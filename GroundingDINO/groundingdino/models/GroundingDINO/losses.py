import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from util import box_ops
from .utils import sigmoid_focal_loss   

from scipy.optimize import linear_sum_assignment

import os
import copy

from torchvision.ops import roi_pool
import torch.distributed as dist 

from .matcher import search_query_pos

import math
import numpy as np
import random
import bisect


    
def padding_last(logits, max_len=2048):
    logits_pad = torch.full((*logits.shape[:-1], max_len), 
                               float("-inf"), device=logits.device)
    logits_pad[..., : logits.shape[-1]] = logits

    return logits_pad

def shrink_sigmoid(x, scale=1.):
    return 1.0 / (1.0 + torch.exp(-scale * x))


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x

def is_distributed():
    return dist.is_available() and dist.is_initialized()


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)


        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes,
                   object_token, relation_token, rel_text_dict, 
                   rel_text_dict_t=None, outputs_t=None, indices_t=None):
        """
          compute loss for relations 
        """
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        device = outputs['pred_logits'].device
        bs, num_queries = outputs['pred_logits'].shape[:2]
        relation_wo_labels = 'edges' not in targets[0]
        relation_feature_t = None
        relation_is_str = False
        sample_cross_frame = False
        rel_tgt = []
        # check if predicates are str
        for target in targets:
            if 'edges' not in target:
                break

            if len(target['edges']) > 0 and isinstance(target['edges'][0][2], str) and (not relation_is_str):
                relation_is_str = True
                sample_cross_frame = relation_is_str
                break

        # if "edges" provided
        if not relation_wo_labels: # 
            all_edge_lbl = []
            freq_dist = []
            batch_ids = []

            sid, oid = [], []
            s_labels, o_labels = [], []
            sid_t, oid_t = [], []
            for bid, target in enumerate(targets):
                tgt_edges = target['edges']
                if len(tgt_edges) == 0:
                    continue 

                matched = {}
                for src, dst in zip(indices[bid][0].tolist(), indices[bid][1].tolist()):  # src: query index. dst: gt box index.
                    matched[dst] = src
                # negatives
                num_pos = num_total = 0
                n = len(matched)
                full_adj = torch.ones((n,n))-torch.diag(torch.ones(n))
                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:
                        full_adj[edge[0], edge[1]] = 0

                # teacher's matched nodes (optional)
                if outputs_t is not None:
                    matched_t = {}
                    for src, dst in zip(indices_t[bid][0].tolist(), indices_t[bid][1].tolist()):
                        matched_t[dst] = src

                # positives 
                if not self.is_closed_set and self.fix_rel_batch:
                    pos_num_per_img = int(self.rel_batch_per_image * 0.25)
                    if len(tgt_edges) > pos_num_per_img:
                        if isinstance(tgt_edges, list):
                            tgt_edges = random.sample(tgt_edges, pos_num_per_img)
                        else:
                            tgt_edges = tgt_edges[torch.randperm(len(tgt_edges))][:pos_num_per_img]

                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:  # matched: key-gt_boxes | value-pred_boxes (indices)
                        sid.append([bid, matched[edge[0]]])
                        oid.append([bid, matched[edge[1]]])
                        if relation_is_str:
                            rel_tgt.append(edge[2])
                        else:
                            all_edge_lbl.append(edge[2])

                        num_pos += 1
                        #full_adj[edge[0], edge[1]] = 0

                        if self.rln_freq_bias is not None:
                            s_labels.append(target['labels'][edge[0]])
                            o_labels.append(target['labels'][edge[1]])

                        if outputs_t is not None:
                            sid_t.append([bid, matched_t[edge[0]]])
                            oid_t.append([bid, matched_t[edge[1]]])

                # negatives 
                neg_edges = torch.nonzero(full_adj)
                if self.is_closed_set or (relation_is_str and not self.fix_rel_batch):
                    num_neg_per_img = max(1, num_pos * 3)  
                else:
                    num_neg_per_img = max(1, self.rel_batch_per_image - num_pos) 

                if len(neg_edges) >= num_neg_per_img:
                    if outputs_t is not None and not self.unsupervised_distill:
                        # sample
                        hs_obj_t = outputs_t['hs_obj'][bid]
                        nsid, noid = [], []
                        for edge in neg_edges.tolist():
                            nsid.append(matched_t[edge[0]])
                            noid.append(matched_t[edge[1]])
                        nsid = torch.as_tensor(nsid)
                        noid = torch.as_tensor(noid)

                        feat = torch.cat((hs_obj_t[nsid], hs_obj_t[noid],
                                          outputs_t['hs_rln'][bid].flatten(1).repeat(nsid.shape[0], 1)),
                                          1)
                        with torch.no_grad():
                            feat = self.rln_proj_teacher(feat)
                            encoded_text = rel_text_dict_t['encoded_text'][bid]
                            feat = (feat @ encoded_text.T).sigmoid()

                        feat_score = feat.max(-1)[0]

                        if self.rel_proposals_threshold_enabled:
                            keep1 = torch.where(feat_score > self.rel_proposals_threshold)[0]
                            keep2 = feat_score.topk(num_neg_per_img)[1]
                            keep = torch.cat((keep1, keep2)).unique()
                        else:
                            keep = feat_score.topk(num_neg_per_img)[1]

                        neg_edges = neg_edges[keep.to(neg_edges.device)]
                    else:
                        if sample_cross_frame:
                            idx_ = torch.randperm(neg_edges.shape[0])[: max(1, num_pos*3)]
                            neg_edges = neg_edges[idx_]
                        else:
                            idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                            neg_edges = neg_edges[idx_]

                for edge in neg_edges.tolist():
                    sid.append([bid, matched[edge[0]]])
                    oid.append([bid, matched[edge[1]]])

                    if self.rln_freq_bias is not None:
                        s_labels.append(target['labels'][edge[0]])
                        o_labels.append(target['labels'][edge[1]])

                    if relation_is_str:
                        rel_tgt.append('[UNK]')
                    else:
                        all_edge_lbl.append(0)

                    num_total += 1
                    if outputs_t is not None:
                        sid_t.append([bid, matched_t[edge[0]]])
                        oid_t.append([bid, matched_t[edge[1]]])

                # sample within a batch if needed
                if sample_cross_frame:
                    sample_num = num_neg_per_img - len(neg_edges)
                    if sample_num > 0 and len(targets) > 1:
                        p = np.array([1]*len(targets))
                        p[bid] = 0
                        p = p / p.sum() 
                        sample_bid = np.random.choice(range(len(targets)), sample_num, p=p)
                        sample_tid = np.random.choice(range(object_token.shape[1]), sample_num)

                        for c_bid, c_tid in zip(sample_bid, sample_tid):
                            if len(tgt_edges) > 0:
                                 pos_edge = random.choice(tgt_edges)
                                 pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                            else: 
                                 if len(matched) > 0:
                                     pos_edge = np.random.choice(list(matched.keys()), 2)
                                     pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                                 else:
                                     pos_edge = np.random.choice(range(object_token.shape[1]), 2)

                            if random.random() > 0.5:
                                 sid.append([bid, pos_edge[0]])
                                 oid.append([c_bid, c_tid])
                            else:
                                 oid.append([bid, pos_edge[1]])
                                 sid.append([c_bid, c_tid])
                            rel_tgt.append('[UNK]')
                            num_total += 1 

                num_total += num_pos
                batch_ids.extend([bid] * num_total)

            assert len(sid) == len(oid) and len(sid) > 0, " Error: len(sid):%s, len(oid):%s" %(len(sid), len(oid))
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            if not relation_is_str:
                all_edge_lbl = torch.as_tensor(all_edge_lbl)

            batch_ids = torch.as_tensor(batch_ids)
            if self.ablation_mode == 'wo_rln':
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]]), 1)
            elif self.ablation_mode == 'avg_rln':
                relation_feature = 0
                for k in range(relation_token.shape[1]):
                    relation_feature += self.rln_proj(torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids][:, k, :]), 1))

                relation_feature /= relation_token.shape[1]
            else:
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],  # sid: (n, 2). (n, 0):batch (n, 1):query_index
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids].flatten(1)), 1)

            if self.rln_freq_bias is not None:
                s_labels = torch.as_tensor(s_labels).to(object_token.device)
                o_labels = torch.as_tensor(o_labels).to(object_token.device)
                freq_dist.append(self.rln_freq_bias( \
                            torch.stack((s_labels, o_labels), 1)
                        ))
            if outputs_t is not None:
                sid_t = torch.as_tensor(sid_t)
                oid_t = torch.as_tensor(oid_t)
                relation_feature_t = torch.cat((outputs_t['hs_obj'][sid_t[:, 0], sid_t[:, 1]],
                                                outputs_t['hs_obj'][oid_t[:, 0], oid_t[:, 1]],
                                                outputs_t['hs_rln'][batch_ids].flatten(1)), 1)

        else: # open-set
            rel_tgt = []
            batch_ids = []
            sid, oid = [], []

            for bid, target in enumerate(targets):
                if len(target['relations']) == 0:
                    continue

                grounded = {}
                cur_num = 0
                for src, dst in zip(indices[bid][0], indices[bid][1]):
                    grounded[dst.item()] = src.item()

                q_nouns = target['gt_names']
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])

                    rel_tgt.append(rel[2])
                    sid.append([bid, grounded[si]])
                    oid.append([bid, grounded[oi]])
                    cur_num += 1
                    
                # random sample negatives
                n = len(q_nouns)
                full_adj = torch.ones((n, n))-torch.diag(torch.ones(n))
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])
                    full_adj[si, oi] = 0
                neg_edges = torch.nonzero(full_adj)

                num_neg_per_img = max(1, len(target['relations']) * 3)
                if len(neg_edges) >= num_neg_per_img:
                    idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                    neg_edges = neg_edges[idx_,:]

                for item in neg_edges.tolist():
                    sid.append([bid, grounded[item[0]]])
                    oid.append([bid, grounded[item[1]]])
                    rel_tgt.append('[UNK]')
                    cur_num += 1

                # random sampling from other images in a batch 
                if len(neg_edges) < num_neg_per_img: 
                    for rel in target['relations']:
                        if rel[0] not in q_nouns or rel[1] not in q_nouns:
                            continue 
                        si = q_nouns.index(rel[0])
                        oi = q_nouns.index(rel[1])

                        next_id = bid
                        assert len(targets) >= 1, "batch size must be greather than 1!"
                        if len(targets) == 1:
                            continue

                        while next_id == bid:
                            next_id = random.randint(0, len(targets)-1)

                        rid = random.randint(0, object_token.shape[1]-1)
                        if random.randint(0, 1) == 0:
                            sid.append([bid, grounded[si]])
                            oid.append([next_id,  rid])
                        else:
                            sid.append([next_id, rid])
                            oid.append([bid, grounded[oi]])

                        rel_tgt.append('[UNK]')
                        cur_num += 1

                batch_ids.extend([bid] * cur_num)

            batch_ids = torch.as_tensor(batch_ids)
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                          object_token[oid[:, 0], oid[:, 1]],
                                          relation_token[batch_ids].flatten(1)
                                          ), 1)

        assert len(relation_feature) > 0, "No relation features !"

        # random permute
        _idx_ = torch.randperm(len(relation_feature))
        batch_ids = batch_ids[_idx_]
        relation_feature = relation_feature[_idx_]
        if self.ablation_mode == 'avg_rln':
            pass
        else:
            relation_feature = self.rln_proj(relation_feature)

        if relation_feature_t is not None:
            with torch.no_grad():
                relation_feature_t = self.rln_proj_teacher(relation_feature_t[_idx_])

        if (not relation_wo_labels) and (not relation_is_str):
            all_edge_lbl = all_edge_lbl[_idx_]
        else:
            rel_tgt = [rel_tgt[e] for e in _idx_.tolist()]

        # loss 
        if self.rln_classifier is not None:
            assert not relation_wo_labels, "rln_classifier should be None for open vocabulary !"
            rel_logits = self.rln_classifier(relation_feature)
            if self.rln_freq_bias is not None:
                freq_dist = torch.cat(freq_dist, 0)[_idx_]
                rel_logits += freq_dist



            rel_tgt_onehot = torch.zeros([rel_logits.shape[0], rel_logits.shape[1]],
                                          dtype =rel_logits.dtype, 
                                          layout=rel_logits.layout, 
                                          device=rel_logits.device)

            all_edge_lbl = all_edge_lbl.to(rel_logits.device)
            rel_tgt_onehot.scatter_(-1, all_edge_lbl.unsqueeze(-1), 1)
        else:
            encoded_text = rel_text_dict['encoded_text'][batch_ids]
            text_mask = rel_text_dict['text_token_mask'][batch_ids]
            input_ids = rel_text_dict['input_ids'][batch_ids]


            rel_logits  = torch.einsum("a d, a b d -> a b", relation_feature, encoded_text)
            rel_logits.masked_fill_(~text_mask, float('-inf'))
            # padding to max_text_len
            rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
            rel_tgt_onehot = torch.zeros_like(rel_logits)

            if relation_feature_t is not None and not self.unsupervised_distill:
                with torch.no_grad():
                    encoded_text_t = rel_text_dict_t['encoded_text'][batch_ids]
                    rel_logits_t  = torch.einsum("a d, a b d -> a b", relation_feature_t, encoded_text_t)
                    rel_logits_t.masked_fill_(~text_mask, float('-inf'))
                    rel_logits_t = padding_last(rel_logits_t, rel_logits.shape[-1])

                rel_logits_t = shrink_sigmoid(rel_logits_t, 2.0) # suitable for OvR
                #rel_logits_t.sigmoid_()

            if relation_wo_labels or relation_is_str:
                for ii, name in enumerate(rel_tgt):
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name + '.').input_ids[1:-1]
                    all_ids = input_ids[ii]

                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids: {}".format(
                            name, self.tokenizer.decode(all_ids))
                    assert start_i < rel_tgt_onehot.shape[1] and end_i < rel_tgt_onehot.shape[1], "start_i:{}, end_i:{}, tgt shape:{}".format(start_i, end_i, rel_tgt_onehot.shape)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0

            else:
                for ii, label in enumerate(all_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]' and relation_feature_t is not None and not self.unsupervised_distill: # use teacher's output
                        rel_tgt_onehot[ii] = rel_logits_t[ii]
                        continue 

                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    all_ids = input_ids[ii]
                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0



        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

                
        if not self.focal_loss_for_edges:  # CE
            loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
            losses = dict(loss_edges=loss)
        else: # focal loss
            alpha, gamma = 0.25, 2.0
            eps = 1e-5 

            # if not torch.isfinite(rel_logits).all():
            #     bad = ~torch.isfinite(rel_logits)
            #     print("rel_logits has NaN/Inf")
            #     print("num bad:", bad.sum().item())
            #     print("max:", rel_logits.max().item())
            #     print("min:", rel_logits.min().item())
            #     assert False

            rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
            rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

            # if (rel_prob >= 1.0).any():
            #     print("rel_prob >= 1 detected")
            #     print("max prob:", rel_prob.max().item())
            #     idx = torch.where(rel_prob >= 1.0)
            #     print("example logits:", rel_logits[idx][:5])
            #     assert False

            # tmp = 1.0 - rel_prob
            # if (tmp <= 0).any():
            #     print("1 - rel_prob <= 0 detected")
            #     print("min value:", tmp.min().item())
            #     idx = torch.where(tmp <= 0)
            #     print("prob:", rel_prob[idx][:5])
            #     print("logits:", rel_logits[idx][:5])
            #     assert False

            # log_term = torch.log(1.0 - rel_prob)
            # if not torch.isfinite(log_term).all():
            #     bad = ~torch.isfinite(log_term)
            #     print("log(1-rel_prob) produced inf/nan")
            #     print("bad count:", bad.sum().item())
            #     print("example prob:", rel_prob[bad][:5])
            #     assert False

            rel_mask = (rel_logits != float('-inf')).float()
          
            rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
            rel_weight[rel_weight == 0] = 1.0

            pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
            neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

            pos_loss = pos_loss.sum() 
            neg_loss = neg_loss.sum() 


            loss = (pos_loss + neg_loss) / rel_num

            losses = dict(loss_edges=loss)
            with torch.no_grad():
                losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
                losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

            if outputs_t is not None and self.unsupervised_distill:
                ids = torch.where(all_edge_lbl == 0)[0]
                loss_distill = F.l1_loss(relation_feature[ids], relation_feature_t[ids], reduction='sum') / rel_num

                losses['loss_distill'] = loss_distill


        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)
        if os.environ.get("DEBUG") == '1':
            import pdb; pdb.set_trace()

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]


        self.global_iter = global_iter



        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {'object_token': outputs['hs_obj'],
                          'relation_token': outputs['hs_rln'],
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t
                    kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {'object_token': aux_outputs['hs_obj'],
                                  'relation_token': aux_outputs['hs_rln'],
                                  'rel_text_dict': outputs.get('rel_text_dict', None),
                                  'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                                 }
                        if outputs_t is not None:
                            kwargs['outputs_t'] = aux_outputs_t
                            kwargs['indices_t'] = indices_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {'object_token': interm_outputs['hs_obj'],
                              'relation_token': interm_outputs['hs_rln'],
                              'rel_text_dict': outputs.get('rel_text_dict', None), 
                              'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                             }
                    if outputs_t is not None:
                        kwargs['outputs_t'] = interm_outputs_t
                        kwargs['indices_t'] = indices_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class DRMSetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)


        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        # self.ind_to_classes = kwargs.get("ind_to_classes", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes,
                   object_token, relation_token, rel_text_dict, 
                   rel_text_dict_t=None, outputs_t=None, indices_t=None):
        """
          compute loss for relations 
        """
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        device = outputs['pred_logits'].device
        bs, num_queries = outputs['pred_logits'].shape[:2]
        relation_wo_labels = 'edges' not in targets[0]
        relation_feature_t = None
        relation_is_str = False
        sample_cross_frame = False
        rel_tgt = []
        # check if predicates are str
        for target in targets:
            if 'edges' not in target:
                break

            if len(target['edges']) > 0 and isinstance(target['edges'][0][2], str) and (not relation_is_str):
                relation_is_str = True
                sample_cross_frame = relation_is_str
                break

        # ------------------- sample triplet -------------------
        # if "edges" provided
        if not relation_wo_labels: # 
            all_edge_lbl = []
            freq_dist = []
            batch_ids = []

            sid, oid = [], []
            s_labels, o_labels = [], []
            sid_t, oid_t = [], []

            pair_indexs = []
            pair_indexs_t = []
            triplet_labels = []

            for bid, target in enumerate(targets):
                tgt_edges = target['edges']
                if len(tgt_edges) == 0:
                    continue 

                matched = {}
                for src, dst in zip(indices[bid][0].tolist(), indices[bid][1].tolist()):  # src: query index. dst: gt box index.
                    matched[dst] = src
                # negatives
                num_pos = num_total = 0
                n = len(matched)
                full_adj = torch.ones((n,n))-torch.diag(torch.ones(n))
                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:
                        full_adj[edge[0], edge[1]] = 0

                # teacher's matched nodes (optional)
                if outputs_t is not None:
                    matched_t = {}
                    for src, dst in zip(indices_t[bid][0].tolist(), indices_t[bid][1].tolist()):
                        matched_t[dst] = src

                
                # positives 
                if not self.is_closed_set and self.fix_rel_batch:
                    pos_num_per_img = int(self.rel_batch_per_image * 0.25)
                    if len(tgt_edges) > pos_num_per_img:
                        if isinstance(tgt_edges, list):
                            tgt_edges = random.sample(tgt_edges, pos_num_per_img)
                        else:
                            tgt_edges = tgt_edges[torch.randperm(len(tgt_edges))][:pos_num_per_img]

                i_pair_index = []
                i_pair_index_t = []
                i_subject_labels = []
                i_object_labels = []
                i_predicate_labels = []
                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:  # matched: key-gt_boxes | value-pred_boxes (indices)
                        sid.append([bid, matched[edge[0]]])
                        oid.append([bid, matched[edge[1]]])

                        i_pair_index.append(torch.tensor([matched[edge[0]], matched[edge[1]]]).to(device))
                        i_subject_labels.append(target['labels'][edge[0]])
                        i_object_labels.append(target['labels'][edge[1]])
                        i_predicate_labels.append(edge[2])

                        if relation_is_str:
                            rel_tgt.append(edge[2])
                        else:
                            all_edge_lbl.append(edge[2])

                        num_pos += 1
                        #full_adj[edge[0], edge[1]] = 0

                        if self.rln_freq_bias is not None:
                            s_labels.append(target['labels'][edge[0]])
                            o_labels.append(target['labels'][edge[1]])

                        if outputs_t is not None:
                            sid_t.append([bid, matched_t[edge[0]]])
                            oid_t.append([bid, matched_t[edge[1]]])

                            i_pair_index_t.append(torch.tensor([matched_t[edge[0]], matched_t[edge[1]]]).to(device))

                # negatives 
                neg_edges = torch.nonzero(full_adj)
                if self.is_closed_set or (relation_is_str and not self.fix_rel_batch):
                    num_neg_per_img = max(1, num_pos * 3)  
                else:
                    num_neg_per_img = max(1, self.rel_batch_per_image - num_pos) 

                if len(neg_edges) >= num_neg_per_img:
                    if outputs_t is not None and not self.unsupervised_distill:
                        # sample
                        hs_obj_t = outputs_t['hs_obj'][bid]
                        nsid, noid = [], []
                        for edge in neg_edges.tolist():
                            nsid.append(matched_t[edge[0]])
                            noid.append(matched_t[edge[1]])
                        nsid = torch.as_tensor(nsid)
                        noid = torch.as_tensor(noid)
                        
                        feat = torch.cat((hs_obj_t[nsid], hs_obj_t[noid],
                                          outputs_t['hs_rln'][bid].flatten(1).repeat(nsid.shape[0], 1)),
                                          1)
                        with torch.no_grad():
                            feat = self.rln_proj_teacher(feat)
                            encoded_text = rel_text_dict_t['encoded_text'][bid]
                            feat = (feat @ encoded_text.T).sigmoid()

                        feat_score = feat.max(-1)[0]

                        if self.rel_proposals_threshold_enabled:
                            keep1 = torch.where(feat_score > self.rel_proposals_threshold)[0]
                            keep2 = feat_score.topk(num_neg_per_img)[1]
                            keep = torch.cat((keep1, keep2)).unique()
                        else:
                            keep = feat_score.topk(num_neg_per_img)[1]

                        neg_edges = neg_edges[keep.to(neg_edges.device)]
                    else:
                        if sample_cross_frame:
                            idx_ = torch.randperm(neg_edges.shape[0])[: max(1, num_pos*3)]
                            neg_edges = neg_edges[idx_]
                        else:
                            idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                            neg_edges = neg_edges[idx_]
                
                for edge in neg_edges.tolist():
                    sid.append([bid, matched[edge[0]]])
                    oid.append([bid, matched[edge[1]]])

                    i_pair_index.append(torch.tensor([matched[edge[0]], matched[edge[1]]]).to(device))
                    i_subject_labels.append(target['labels'][edge[0]])
                    i_object_labels.append(target['labels'][edge[1]])
                    i_predicate_labels.append(0)

                    if self.rln_freq_bias is not None:
                        s_labels.append(target['labels'][edge[0]])
                        o_labels.append(target['labels'][edge[1]])

                    if relation_is_str:
                        rel_tgt.append('[UNK]')
                    else:
                        all_edge_lbl.append(0)

                    num_total += 1
                    if outputs_t is not None:
                        sid_t.append([bid, matched_t[edge[0]]])
                        oid_t.append([bid, matched_t[edge[1]]])

                        i_pair_index_t.append(torch.tensor([matched_t[edge[0]], matched_t[edge[1]]]).to(device))

                i_pair_index = torch.stack(i_pair_index)
                if outputs_t is not None:
                    i_pair_index_t = torch.stack(i_pair_index_t)
                i_subject_labels = torch.stack(i_subject_labels)
                i_object_labels = torch.stack(i_object_labels)
                i_predicate_labels = torch.tensor(i_predicate_labels).to(device)

                pair_indexs.append(i_pair_index)
                if outputs_t is not None:
                    pair_indexs_t.append(i_pair_index_t)
                triplet_labels.append(torch.stack([i_subject_labels, i_object_labels, i_predicate_labels], dim=1))

                # sample within a batch if needed
                if sample_cross_frame:
                    sample_num = num_neg_per_img - len(neg_edges)
                    if sample_num > 0 and len(targets) > 1:
                        p = np.array([1]*len(targets))
                        p[bid] = 0
                        p = p / p.sum() 
                        sample_bid = np.random.choice(range(len(targets)), sample_num, p=p)
                        sample_tid = np.random.choice(range(object_token.shape[1]), sample_num)
                         
                        for c_bid, c_tid in zip(sample_bid, sample_tid):
                            if len(tgt_edges) > 0:
                                 pos_edge = random.choice(tgt_edges)
                                 pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                            else: 
                                 if len(matched) > 0:
                                     pos_edge = np.random.choice(list(matched.keys()), 2)
                                     pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                                 else:
                                     pos_edge = np.random.choice(range(object_token.shape[1]), 2)

                            if random.random() > 0.5:
                                 sid.append([bid, pos_edge[0]])
                                 oid.append([c_bid, c_tid])
                            else:
                                 oid.append([bid, pos_edge[1]])
                                 sid.append([c_bid, c_tid])
                            rel_tgt.append('[UNK]')
                            num_total += 1 

                num_total += num_pos
                batch_ids.extend([bid] * num_total)

            assert len(sid) == len(oid) and len(sid) > 0, " Error: len(sid):%s, len(oid):%s" %(len(sid), len(oid))
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            if not relation_is_str:
                all_edge_lbl = torch.as_tensor(all_edge_lbl)

            batch_ids = torch.as_tensor(batch_ids)
            if self.ablation_mode == 'wo_rln':
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]]), 1)
            elif self.ablation_mode == 'avg_rln':
                relation_feature = 0
                for k in range(relation_token.shape[1]):
                    relation_feature += self.rln_proj(torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids][:, k, :]), 1))

                relation_feature /= relation_token.shape[1]
            else:
                # here
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],  # object_token: (B, 900, 256)
                                 object_token[oid[:, 0], oid[:, 1]],
                                 relation_token[batch_ids].flatten(1)), 1)

            if self.rln_freq_bias is not None:
                s_labels = torch.as_tensor(s_labels).to(object_token.device)
                o_labels = torch.as_tensor(o_labels).to(object_token.device)
                freq_dist.append(self.rln_freq_bias( \
                            torch.stack((s_labels, o_labels), 1)
                        ))
            if outputs_t is not None:
                sid_t = torch.as_tensor(sid_t)
                oid_t = torch.as_tensor(oid_t)
                relation_feature_t = torch.cat((outputs_t['hs_obj'][sid_t[:, 0], sid_t[:, 1]],
                                                outputs_t['hs_obj'][oid_t[:, 0], oid_t[:, 1]],
                                                outputs_t['hs_rln'][batch_ids].flatten(1)), 1)
        else: # open-set
            rel_tgt = []
            batch_ids = []
            sid, oid = [], []

            for bid, target in enumerate(targets):
                if len(target['relations']) == 0:
                    continue

                grounded = {}
                cur_num = 0
                for src, dst in zip(indices[bid][0], indices[bid][1]):
                    grounded[dst.item()] = src.item()

                q_nouns = target['gt_names']
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])

                    rel_tgt.append(rel[2])
                    sid.append([bid, grounded[si]])
                    oid.append([bid, grounded[oi]])
                    cur_num += 1
                    
                # random sample negatives
                n = len(q_nouns)
                full_adj = torch.ones((n, n))-torch.diag(torch.ones(n))
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue 
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])
                    full_adj[si, oi] = 0
                neg_edges = torch.nonzero(full_adj)

                num_neg_per_img = max(1, len(target['relations']) * 3)
                if len(neg_edges) >= num_neg_per_img:
                    idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                    neg_edges = neg_edges[idx_,:]

                for item in neg_edges.tolist():
                    sid.append([bid, grounded[item[0]]])
                    oid.append([bid, grounded[item[1]]])
                    rel_tgt.append('[UNK]')
                    cur_num += 1

                # random sampling from other images in a batch 
                if len(neg_edges) < num_neg_per_img: 
                    for rel in target['relations']:
                        if rel[0] not in q_nouns or rel[1] not in q_nouns:
                            continue 
                        si = q_nouns.index(rel[0])
                        oi = q_nouns.index(rel[1])

                        next_id = bid
                        assert len(targets) >= 1, "batch size must be greather than 1!"
                        if len(targets) == 1:
                            continue

                        while next_id == bid:
                            next_id = random.randint(0, len(targets)-1)

                        rid = random.randint(0, object_token.shape[1]-1)
                        if random.randint(0, 1) == 0:
                            sid.append([bid, grounded[si]])
                            oid.append([next_id,  rid])
                        else:
                            sid.append([next_id, rid])
                            oid.append([bid, grounded[oi]])

                        rel_tgt.append('[UNK]')
                        cur_num += 1

                batch_ids.extend([bid] * cur_num)

            batch_ids = torch.as_tensor(batch_ids)
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                          object_token[oid[:, 0], oid[:, 1]],
                                          relation_token[batch_ids].flatten(1)
                                          ), 1)

        assert len(relation_feature) > 0, "No relation features !"

        # # random permute
        # _idx_ = torch.randperm(len(relation_feature))
        # batch_ids = batch_ids[_idx_]
        # relation_feature = relation_feature[_idx_]

        if self.ablation_mode == 'avg_rln':
            pass
        else:
            # and here
            relation_feature = self.rln_proj(relation_feature)
            if outputs_t is not None and self.unsupervised_distill:
                relation_feature_t = self.rln_proj_t(relation_feature_t)

        # ------------------- context layer -------------------
        def prepare_for_drm(indices, targets, object_token, relation_feature, pair_indexs, bert, text_weight_generate_layer):
            entity_visual_features = []
            for i_indices, i_object_token in zip(indices, object_token):  # [0]: query [1]: gt
                entity_visual_features.append(i_object_token[i_indices[0]])  # take the matched hs in indices order

            entity_labels = []
            for i_indices, target in zip(indices, targets):
                entity_labels.append(target['labels'][i_indices[1]])  # take the matched entity labels in indices order
            entity_text_features = self.get_entity_text_feature(entity_labels, bert, device, text_weight_generate_layer)

            query_to_drm_index_map_list = []
            for i_indices in indices:
                query_to_drm_index_map = {}
                for i, query_index in enumerate(i_indices[0]):
                    query_to_drm_index_map[query_index] = i  # build the map in indices order
                query_to_drm_index_map_list.append(query_to_drm_index_map)

            drm_pair_indexs = []
            for i_pair_indexs, i_query_to_drm_index_map in zip(pair_indexs, query_to_drm_index_map_list):
                max_key = max(i_query_to_drm_index_map.keys())
                lookup_table = torch.full((max_key + 1,), -1, dtype=i_pair_index.dtype)
                for k, v, in i_query_to_drm_index_map.items():
                    lookup_table[k] = v
                drm_pair_indexs.append(lookup_table[i_pair_indexs])

            relation_num_list = [len(x) for x in pair_indexs]
            union_visual_features = list(relation_feature.split(relation_num_list, dim=0))

            return drm_pair_indexs, entity_visual_features, union_visual_features, entity_text_features

        pair_indexs, entity_visual_features, union_visual_features, entity_text_features = prepare_for_drm(indices, targets, object_token, relation_feature, pair_indexs, self.bert, self.text_weight_generate_layer)
        if outputs_t is not None and self.unsupervised_distill:
            pair_indexs_t, entity_visual_features_t, union_visual_features_t, entity_text_features_t = prepare_for_drm(indices_t, targets, outputs_t['hs_obj'], relation_feature_t, pair_indexs_t, self.bert_t, self.text_weight_generate_layer_t)

        for i in range(len(triplet_labels)):
            perm = torch.randperm(len(triplet_labels[i]))
            triplet_labels[i] = triplet_labels[i][perm]
            pair_indexs[i] = pair_indexs[i][perm]
            union_visual_features[i] = union_visual_features[i][perm]
            if outputs_t is not None and self.unsupervised_distill:
                pair_indexs_t[i] = pair_indexs_t[i][perm]
                union_visual_features_t[i] = union_visual_features_t[i][perm]

        result_dict = self.context_layer(
            pair_indexs=pair_indexs,
            entity_visual_features=entity_visual_features,
            union_visual_features=union_visual_features,
            entity_textual_features=entity_text_features
        )

        if outputs_t is not None and self.unsupervised_distill:
            result_dict_t = self.context_layer_t(
                pair_indexs=pair_indexs_t,
                entity_visual_features=entity_visual_features_t,
                union_visual_features=union_visual_features_t,
                entity_textual_features=entity_text_features_t
            )
        else:
            result_dict_t = None

        if outputs_t is not None and self.unsupervised_distill:
            pass

        # ------------------- drm special loss calculation -------------------
        result_dict['relation_text_feature'] = rel_text_dict['encoded_text']
        result_dict['text_token_mask'] = rel_text_dict['text_token_mask']
        result_dict['input_ids'] = rel_text_dict['input_ids']
        result_dict['triplet_labels'] = triplet_labels
        result_dict['pair_num_list'] = [len(x) for x in pair_indexs]

        if outputs_t is not None:
            result_dict_t['pair_num_list'] = [len(x) for x in pair_indexs]

        loss_dict = self.drm_loss(
            result_dict=result_dict,
            device=device,
            teacher_result_dict=result_dict_t
        )

        # ------------------- end here -------------------
        return loss_dict

    def drm_loss(self, result_dict, device, teacher_result_dict=None):
        gt_triplet_labels = result_dict['triplet_labels']
        gt_triplet_labels = torch.cat(gt_triplet_labels)
        negative_mask = gt_triplet_labels[:, 2] == 0  # True for negative

        # ------------------------ contrastive loss ------------------------
        predicate_label = gt_triplet_labels[:, 2].unsqueeze(1)  # (B * m_i, 1)
        predicate_mask = torch.eq(predicate_label, predicate_label.T)  # (B * m_i, B * m_i), same category as True
        predicate_mask.fill_diagonal_(False)  # Self-contrast as False
        predicate_mask[:, negative_mask] = False  # Negative sample as False
        predicate_valid_mask = torch.any(predicate_mask, dim=-1) & (~ negative_mask)  # Remain those have at least one same sample and its not negative sample itself.
        # predicate_valid_mask = torch.any(predicate_mask, dim=-1)
        predicate_mask = predicate_mask[predicate_valid_mask]

        # (B * n', 1, 3) == (1, B * n', 3)
        triplet_mask = (gt_triplet_labels[:, None, :] == gt_triplet_labels[None, :, :]).all(-1)  # same category as True
        triplet_mask.fill_diagonal_(False)
        triplet_mask[:, negative_mask] = False
        triplet_valid_mask = torch.any(triplet_mask, dim=-1) & (~ negative_mask)
        triplet_mask = triplet_mask[triplet_valid_mask]  # At least one same sample and not padded.

        (
            projected_predicate_gated_visual_features,
            projected_predicate_features,
            projected_triplet_gated_visual_features,
            projected_triplet_features,
            _,
            _
        ) = self.get_flattened_result(result_dict)
        predicate_gated_visual_loss = self.i_contrastive(projected_predicate_gated_visual_features, predicate_mask, predicate_valid_mask, device, tau=0.2)  # 0.2
        predicate_loss = self.i_contrastive(projected_predicate_features, predicate_mask, predicate_valid_mask, device, tau=0.2)  # 0.2
        triplet_gated_visual_loss = self.i_contrastive(projected_triplet_gated_visual_features, triplet_mask, triplet_valid_mask, device, tau=0.1)  # 0.1
        triplet_loss = self.i_contrastive(projected_triplet_features, triplet_mask, triplet_valid_mask, device, tau=0.1)  # 0.1

        # ------------------------ classification loss ------------------------
        predicate_feature = result_dict['predicate_combined_features_in_text_space']
        triplet_feature = result_dict['triplet_combined_features_in_text_space']
        relation_text_feature = result_dict['relation_text_feature']
        text_mask = result_dict['text_token_mask']
        triplet_labels = result_dict['triplet_labels']
        input_ids = result_dict['input_ids']

        predicate_rel_logits = []
        triplet_rel_logits = []
        rel_tgt_onehot = []
        for (
            i_predicate_feature,
            i_triplet_feature,
            i_relation_text_feature,
            pair_num,
            i_text_mask,
            i_triplet_labels,
            i_input_ids
            ) in zip(
                predicate_feature,
                triplet_feature,
                relation_text_feature,
                result_dict['pair_num_list'],
                text_mask,
                triplet_labels,
                input_ids
                ):
            i_predicate_feature = i_predicate_feature[:pair_num]
            i_triplet_feature = i_triplet_feature[:pair_num]
            i_predicate_rel_logits = i_predicate_feature @ i_relation_text_feature.T
            i_triplet_rel_logits = i_triplet_feature @ i_relation_text_feature.T
            i_predicate_rel_logits.masked_fill_(~i_text_mask, float('-inf'))
            i_triplet_rel_logits.masked_fill_(~i_text_mask, float('-inf'))
            i_predicate_rel_logits = padding_last(i_predicate_rel_logits, 512)
            i_triplet_rel_logits = padding_last(i_triplet_rel_logits, 512)

            rel_tgt = i_triplet_labels[:, 2]
            i_rel_tgt_onehot = torch.zeros_like(i_predicate_rel_logits)
            for i, label in enumerate(rel_tgt):
                name = self.ind_to_predicates[label]
                if name == '[UNK]':
                    continue
                ids = self.tokenizer(name + '.').input_ids[1:-1]
                all_ids = i_input_ids
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)
                i_rel_tgt_onehot[i, start_i: end_i] = 1.0

            predicate_rel_logits.append(i_predicate_rel_logits)
            triplet_rel_logits.append(i_triplet_rel_logits)
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        predicate_rel_logits = torch.cat(predicate_rel_logits)
        triplet_rel_logits = torch.cat(triplet_rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)

        # rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(rel_num)
        # rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device).item()

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 
        predicate_rel_prob = predicate_rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
        triplet_rel_prob = triplet_rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
        rel_mask = (predicate_rel_logits != float('-inf')).float()

        rel_prob = (predicate_rel_prob + triplet_rel_prob) / 2

        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum() 
        neg_loss = neg_loss.sum() 

        loss = (pos_loss + neg_loss) / rel_num

        losses = dict(loss_edges=loss)
        with torch.no_grad():
            losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
            losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

        # relation_category_text_features_num_list = []
        # relation_label_to_feature_map = {}
        # start_pos = 0
        # for i, i_relation_text_features in enumerate(relation_text_features_list):
        #     i_relation_category_text_features_length = i_relation_text_features.shape[0]
        #     relation_category_text_features_num_list.append(i_relation_category_text_features_length)
        #     relation_label_to_feature_map[i] = [start_pos, start_pos + i_relation_category_text_features_length]
        #     start_pos = start_pos + i_relation_category_text_features_length
        # relation_text_features = torch.cat(relation_text_features_list, dim=0)

        # predicate_logits = predicate_combined_features_in_text_space @ relation_text_features.T
        # triplet_logits = triplet_combined_features_in_text_space @ relation_text_features.T
        # predicate_probability = (predicate_logits.sigmoid() + triplet_logits.sigmoid()) / 2.0  # (B * m_i, num_classes)
        # predicate_probability = torch.clamp(predicate_probability, 1e-8, 1 - 1e-8)

        # predicate_label_one_hot = []
        # for i_label in gt_triplet_labels[:, 2]:
        #     i_one_hot = torch.zeros(1, relation_text_features.shape[0]).float().to(device)
        #     if i_label != 0:
        #         mapped_index = (i_label == relation_labels).nonzero(as_tuple=True)[0].item()
        #         i_one_hot[0][relation_label_to_feature_map[mapped_index][0]: relation_label_to_feature_map[mapped_index][1]] = 1.0
        #     predicate_label_one_hot.append(i_one_hot)
        # predicate_label_one_hot = torch.cat(predicate_label_one_hot, dim=0)

        # predicate_weight = (predicate_label_one_hot > 0.5).sum(1, keepdim=True)  # (B, 1)
        # predicate_weight[predicate_weight == 0] = 1.0

        # pos_loss = - torch.log(predicate_probability) * ((1.0 - predicate_probability) ** 2.0) * predicate_label_one_hot / predicate_weight
        # neg_loss = -torch.log(1.0 - predicate_probability) * (predicate_probability ** 2.0) * (1.0 - predicate_label_one_hot) / predicate_weight
        # pos_loss = pos_loss.sum()
        # neg_loss = neg_loss.sum()
        # classification_loss = (pos_loss + neg_loss) / len(gt_triplet_labels)

        # ------------------------ distillation loss ------------------------
        if teacher_result_dict:
            (
                projected_predicate_gated_visual_features,
                projected_predicate_features,
                projected_triplet_gated_visual_features,
                projected_triplet_features,
                predicate_combined_features_in_text_space,
                triplet_combined_features_in_text_space
            ) = self.get_flattened_result(result_dict)

            (
                teacher_projected_predicate_gated_visual_features,
                teacher_projected_predicate_features,
                teacher_projected_triplet_gated_visual_features,
                teacher_projected_triplet_features,
                teacher_predicate_combined_features_in_text_space,
                teacher_triplet_combined_features_in_text_space
            ) = self.get_flattened_result(teacher_result_dict)

            projected_predicate_gated_visual_features = projected_predicate_gated_visual_features[negative_mask]
            projected_predicate_features = projected_predicate_features[negative_mask]
            projected_triplet_gated_visual_features = projected_triplet_gated_visual_features[negative_mask]
            projected_triplet_features = projected_triplet_features[negative_mask]
            predicate_combined_features_in_text_space = predicate_combined_features_in_text_space[negative_mask]
            triplet_combined_features_in_text_space = triplet_combined_features_in_text_space[negative_mask]

            teacher_projected_predicate_gated_visual_features = teacher_projected_predicate_gated_visual_features[negative_mask]
            teacher_projected_predicate_features = teacher_projected_predicate_features[negative_mask]
            teacher_projected_triplet_gated_visual_features = teacher_projected_triplet_gated_visual_features[negative_mask]
            teacher_projected_triplet_features = teacher_projected_triplet_features[negative_mask]
            teacher_predicate_combined_features_in_text_space = teacher_predicate_combined_features_in_text_space[negative_mask]
            teacher_triplet_combined_features_in_text_space = teacher_triplet_combined_features_in_text_space[negative_mask]

            distillation_loss = 0.0
            distillation_loss += F.l1_loss(projected_predicate_gated_visual_features, teacher_projected_predicate_gated_visual_features, reduction='mean')
            distillation_loss += F.l1_loss(projected_predicate_features, teacher_projected_predicate_features, reduction='mean')
            distillation_loss += F.l1_loss(projected_triplet_gated_visual_features, teacher_projected_triplet_gated_visual_features, reduction='mean')
            distillation_loss += F.l1_loss(projected_triplet_features, teacher_projected_triplet_features, reduction='mean')
            distillation_loss += F.l1_loss(predicate_combined_features_in_text_space, teacher_predicate_combined_features_in_text_space, reduction='mean')
            distillation_loss += F.l1_loss(triplet_combined_features_in_text_space, teacher_triplet_combined_features_in_text_space, reduction='mean')


        losses['loss_predicate_contrastive'] = ((predicate_gated_visual_loss + predicate_loss) / 2.0)
        losses['loss_triplet_contrastive'] = ((triplet_gated_visual_loss + triplet_loss) / 2.0)
        if teacher_result_dict:
            losses['loss_distill'] = distillation_loss

        # loss_dict = {
        #     'loss_predicate_contrastive': ((predicate_gated_visual_loss + predicate_loss) / 2.0),
        #     'loss_triplet_contrastive': ((triplet_gated_visual_loss + triplet_loss) / 2.0),
        #     'loss_predicate_classification': classification_loss
        # }

        return losses

    def get_flattened_result(self, result_dict):
        '''
        each batch's pairs have been padded for attention layer, now conduct de-padding.
        '''
        pair_num_list = result_dict['pair_num_list']

        projected_predicate_gated_visual_features = torch.cat(
            [result_dict['projected_predicate_gated_visual_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
            dim=0
            )
        projected_predicate_features = torch.cat(
            [result_dict['projected_predicate_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
            dim=0
            )
        projected_triplet_gated_visual_features = torch.cat(
            [result_dict['projected_triplet_gated_visual_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
            dim=0
            )
        projected_triplet_features = torch.cat(
            [result_dict['projected_triplet_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
            dim=0
            )
        predicate_combined_features_in_text_space = torch.cat(
            [result_dict['predicate_combined_features_in_text_space'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
            dim=0
            )
        triplet_combined_features_in_text_space = torch.cat(
            [result_dict['triplet_combined_features_in_text_space'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
            dim=0
            )
        return (
            projected_predicate_gated_visual_features,
            projected_predicate_features,
            projected_triplet_gated_visual_features,
            projected_triplet_features,
            predicate_combined_features_in_text_space,
            triplet_combined_features_in_text_space
        )

    def i_contrastive(self, features, mask, valid_mask, device, tau):
        """
        Args:
            features: (num_pairs, d)
            mask: (valid, num_pairs), True for same category.
            valid_mask: (num_pairs,), True for valid samples.
        """
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        source_features = features[valid_mask]  # (valid, d)
        destiny_features = features  # (num_pairs, d)

        logits_source_to_destiny = torch.div(torch.matmul(source_features, destiny_features.T), tau)  # (valid, num_pairs)
        logits_max, _ = torch.max(logits_source_to_destiny, dim=-1, keepdim=True)
        logits = logits_source_to_destiny - logits_max.detach()
        log_probability = torch.nn.functional.log_softmax(logits, dim=-1)  # (valid, num_pairs)
        loss = - torch.sum(mask * log_probability, dim=-1) / torch.sum(mask, dim=-1)
        return loss.mean()

    def get_entity_text_feature(self, entity_labels, bert, device, text_weight_generate_layer):
        """
        Each label (n,) is converted into corresponding category, then concatenated into a single string,
        then fed into bert to produce word embedding (m, 768).

        The word embedding is then cut into pieces to its corresponding word, store in a list.

        A simple attention layer is used to refine each word embedding piece into a fix-size text feature token.
        Args:
            entity_labels_list: (n,)
        Returns:
            refined_category_text_feature: (n, 768)
        """
        def get_one_batch_text_features(one_batch_entity_labels, bert):
            entity_category_list = []
            for label in one_batch_entity_labels:
                entity_category_list.append(self.ind_to_classes[label.item()].strip().lower())

            entity_category_text_feature_list = self.category_string_to_text_feature('. '.join(entity_category_list) + '. ', bert, True, device)

            refined_category_text_feature = []
            for i in range(len(entity_category_text_feature_list)):
                text_weight_logits = text_weight_generate_layer(entity_category_text_feature_list[i])  # (n, 1)
                text_weight = F.softmax(text_weight_logits, dim=1)  # (n, 1)
                i_refined_category_text_feature = (text_weight.T @ entity_category_text_feature_list[i])  #  (1, n) @ (n, 768) -> (1, 768)
                refined_category_text_feature.append(i_refined_category_text_feature)

            return torch.cat(refined_category_text_feature, dim=0)  # (n, 1, 768) -> (n, 768)

        entity_text_features = []
        for i_entity_labels in entity_labels:
            entity_text_features.append(get_one_batch_text_features(i_entity_labels, bert))

        return entity_text_features

    def category_string_to_text_feature(self, category_str, bert, to_piece_list: bool, device):
        '''
        This function takes a string of categories, then convert into encoded embedding if to_piece_list is False.
        otherwise split them into list by its correspoding word.
        Args:
            category_str: Concatenate category string as "apple. banana. orange."
            to_piece_list: whether to split the list or not.
        Returns:
            category_text_features_list: A encoded embedding or a list of word embedding pieces.
        '''
        tokenized_category_list = self.tokenizer(
            category_str,
            padding='longest',
            return_tensors='pt',
        )

        text_self_attention_masks, position_ids = self.generate_masks_with_special_tokens_and_transfer_map(tokenized_category_list, device)

        # tokenized_for_encoder = {k: v.to(self.device) for k, v in tokenized_category_list.items() if k != "attention_mask"}
        # tokenized_for_encoder["attention_mask"] = text_self_attention_masks.to(self.device)
        # tokenized_for_encoder["position_ids"] = position_ids.to(self.device)

        tokenized_for_encoder = {k: v.to(device) for k, v in tokenized_category_list.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks.to(device)
        tokenized_for_encoder["position_ids"] = position_ids.to(device)

        encoded_text = bert(**tokenized_for_encoder)['last_hidden_state'][0]

        if not to_piece_list:
            return encoded_text

        input_ids = tokenized_category_list['input_ids'][0]
        sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]

        category_text_features_list = []
        for i in range(1, len(sep_idx)):
            right_idx = sep_idx[i]
            left_idx = sep_idx[i-1] + 1
            if left_idx >= right_idx:
                continue
            category_text_features_list.append(encoded_text[left_idx:right_idx])

        # if for_category:
        #     assert len(category_text_features_list) == 50
        return category_text_features_list

    def generate_masks_with_special_tokens_and_transfer_map(self, tokenized, device):
        """Generate attention mask between each pair of special tokens
        Args:
            input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
            special_tokens_mask (list): special tokens mask.
        Returns:
            torch.Tensor: attention mask between each special tokens.
        """
        special_tokens_list = [101, 102, 1012, 1029]
        input_ids = tokenized["input_ids"].to(device)
        bs, num_token = input_ids.shape
        # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
        special_tokens_mask = torch.zeros((bs, num_token), device=device).bool()
        for special_token in special_tokens_list:
            special_tokens_mask |= input_ids == special_token

        # idxs: each row is a list of indices of special tokens
        idxs = torch.nonzero(special_tokens_mask)

        # generate attention mask and positional ids
        attention_mask = (
            torch.eye(num_token, device=device).bool().unsqueeze(0).repeat(bs, 1, 1)
        )
        position_ids = torch.zeros((bs, num_token), device=device)
        cate_to_token_mask_list = [[] for _ in range(bs)]
        previous_col = 0
        for i in range(idxs.shape[0]):
            row, col = idxs[i]
            if (col == 0) or (col == num_token - 1):
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
                position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                    0, col - previous_col, device=device
                )
                c2t_maski = torch.zeros((num_token), device=device).bool()
                c2t_maski[previous_col + 1 : col] = True
                cate_to_token_mask_list[row].append(c2t_maski)
            previous_col = col

        cate_to_token_mask_list = [
            torch.stack(cate_to_token_mask_listi, dim=0)
            for cate_to_token_mask_listi in cate_to_token_mask_list
        ]

        # # padding mask
        # padding_mask = tokenized['attention_mask']
        # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

        return attention_mask, position_ids.to(torch.long)

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]


        self.global_iter = global_iter



        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {'object_token': outputs['hs_obj'],
                          'relation_token': outputs['hs_rln'],
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t
                    kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {'object_token': aux_outputs['hs_obj'],
                                  'relation_token': aux_outputs['hs_rln'],
                                  'rel_text_dict': outputs.get('rel_text_dict', None),
                                  'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                                 }
                        if outputs_t is not None:
                            kwargs['outputs_t'] = aux_outputs_t
                            kwargs['indices_t'] = indices_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {'object_token': interm_outputs['hs_obj'],
                              'relation_token': interm_outputs['hs_rln'],
                              'rel_text_dict': outputs.get('rel_text_dict', None), 
                              'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                             }
                    if outputs_t is not None:
                        kwargs['outputs_t'] = interm_outputs_t
                        kwargs['indices_t'] = indices_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups

    def generate_cut_slice_by_caption(self, captions):
        captions = [e[:-1].split('.') for e in captions]
        for caption in captions:
            caption = [('. '.join(e) + '. ') for e in caption]


class DRMOvSGTRSetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)

        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        # self.ind_to_classes = kwargs.get("ind_to_classes", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes, rel_text_dict, rel_text_dict_t=None, outputs_t=None):
        # ------------------- drm special loss calculation -------------------
        result_dict = outputs['context_result_dict']
        result_dict['relation_text_feature'] = rel_text_dict['encoded_text']
        result_dict['text_token_mask'] = rel_text_dict['text_token_mask']
        result_dict['input_ids'] = rel_text_dict['input_ids']
        # result_dict['triplet_labels'] = triplet_labels
        # result_dict['pair_num_list'] = [len(x) for x in pair_indexs]

        result_dict_t = None
        if outputs_t is not None:
            result_dict_t = outputs_t['context_result_dict']

        device = result_dict['predicate_combined_features_in_text_space'].device

        loss_dict = self.drm_loss(
            result_dict=result_dict,
            device=device,
            teacher_result_dict=result_dict_t
        )

        # ------------------- end here -------------------
        return loss_dict

    def drm_loss(self, result_dict, device, teacher_result_dict=None):
        gt_triplet_labels = result_dict['triplet_labels']
        gt_triplet_labels = torch.cat(gt_triplet_labels)
        negative_mask = gt_triplet_labels[:, 2] == 0  # True for negative

        # ------------------------ contrastive loss ------------------------
        predicate_label = gt_triplet_labels[:, 2].unsqueeze(1)  # (B * m_i, 1)
        predicate_mask = torch.eq(predicate_label, predicate_label.T)  # (B * m_i, B * m_i), same category as True
        predicate_mask.fill_diagonal_(False)  # Self-contrast as False
        predicate_mask[:, negative_mask] = False  # Negative sample as False
        predicate_valid_mask = torch.any(predicate_mask, dim=-1) & (~ negative_mask)  # Remain those have at least one same sample and its not negative sample itself.
        # predicate_valid_mask = torch.any(predicate_mask, dim=-1)
        predicate_mask = predicate_mask[predicate_valid_mask]

        # (B * n', 1, 3) == (1, B * n', 3)
        triplet_mask = (gt_triplet_labels[:, None, :] == gt_triplet_labels[None, :, :]).all(-1)  # same category as True
        triplet_mask.fill_diagonal_(False)
        triplet_mask[:, negative_mask] = False
        triplet_valid_mask = torch.any(triplet_mask, dim=-1) & (~ negative_mask)
        triplet_mask = triplet_mask[triplet_valid_mask]  # At least one same sample and not padded.

        (
            projected_predicate_gated_visual_features,
            projected_predicate_features,
            projected_triplet_gated_visual_features,
            projected_triplet_features,
            _,
            _,
            _,
            _
        ) = self.get_flattened_result(result_dict)
        predicate_gated_visual_loss = self.i_contrastive(projected_predicate_gated_visual_features, predicate_mask, predicate_valid_mask, device, tau=0.2)  # 0.2
        predicate_loss = self.i_contrastive(projected_predicate_features, predicate_mask, predicate_valid_mask, device, tau=0.2)  # 0.2
        triplet_gated_visual_loss = self.i_contrastive(projected_triplet_gated_visual_features, triplet_mask, triplet_valid_mask, device, tau=0.1)  # 0.1
        triplet_loss = self.i_contrastive(projected_triplet_features, triplet_mask, triplet_valid_mask, device, tau=0.1)  # 0.1

        # ------------------------ classification loss ------------------------
        predicate_feature = result_dict['predicate_combined_features_in_text_space']
        triplet_feature = result_dict['triplet_combined_features_in_text_space']
        relation_text_feature = result_dict['relation_text_feature']
        text_mask = result_dict['text_token_mask']
        triplet_labels = result_dict['triplet_labels']
        input_ids = result_dict['input_ids']

        predicate_rel_logits = []
        triplet_rel_logits = []
        rel_tgt_onehot = []
        for (
            i_predicate_feature,
            i_triplet_feature,
            i_relation_text_feature,
            pair_num,
            i_text_mask,
            i_triplet_labels,
            i_input_ids
            ) in zip(
                predicate_feature,
                triplet_feature,
                relation_text_feature,
                result_dict['pair_num_list'],
                text_mask,
                triplet_labels,
                input_ids
                ):
            i_predicate_feature = i_predicate_feature[:pair_num]
            i_triplet_feature = i_triplet_feature[:pair_num]
            i_predicate_rel_logits = i_predicate_feature @ i_relation_text_feature.T
            i_triplet_rel_logits = i_triplet_feature @ i_relation_text_feature.T
            i_predicate_rel_logits.masked_fill_(~i_text_mask, float('-inf'))
            i_triplet_rel_logits.masked_fill_(~i_text_mask, float('-inf'))
            i_predicate_rel_logits = padding_last(i_predicate_rel_logits, 512)
            i_triplet_rel_logits = padding_last(i_triplet_rel_logits, 512)

            rel_tgt = i_triplet_labels[:, 2]
            i_rel_tgt_onehot = torch.zeros_like(i_predicate_rel_logits)
            for i, label in enumerate(rel_tgt):
                name = self.ind_to_predicates[label]
                if name == '[UNK]':
                    continue
                ids = self.tokenizer(name + '.').input_ids[1:-1]
                all_ids = i_input_ids
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)
                i_rel_tgt_onehot[i, start_i: end_i] = 1.0

            predicate_rel_logits.append(i_predicate_rel_logits)
            triplet_rel_logits.append(i_triplet_rel_logits)
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        predicate_rel_logits = torch.cat(predicate_rel_logits)
        triplet_rel_logits = torch.cat(triplet_rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)

        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        # rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device).item()

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 
        predicate_rel_prob = predicate_rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
        triplet_rel_prob = triplet_rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
        rel_mask = (predicate_rel_logits != float('-inf')).float()

        rel_prob = (predicate_rel_prob + triplet_rel_prob) / 2

        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum() 
        neg_loss = neg_loss.sum() 

        loss = (pos_loss + neg_loss) / rel_num

        losses = dict(loss_edges=loss)
        with torch.no_grad():
            losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
            losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

        # ------------------------ distillation loss ------------------------
        if teacher_result_dict:
            (
                projected_predicate_gated_visual_features,
                projected_predicate_features,
                projected_triplet_gated_visual_features,
                projected_triplet_features,
                predicate_combined_features_in_text_space,
                triplet_combined_features_in_text_space,
                triplet_labels,
                origin_sub_obj_pair
            ) = self.get_flattened_result(result_dict)

            perm = result_dict['perm']
            (
                teacher_projected_predicate_gated_visual_features,
                teacher_projected_predicate_features,
                teacher_projected_triplet_gated_visual_features,
                teacher_projected_triplet_features,
                teacher_predicate_combined_features_in_text_space,
                teacher_triplet_combined_features_in_text_space,
                teacher_triplet_labels,
                teacher_origin_sub_obj_pair
            ) = self.get_flattened_result(teacher_result_dict, perm)

            assert torch.equal(triplet_labels, teacher_triplet_labels), "Student and teacher triplet label not equal!"
            assert torch.equal(origin_sub_obj_pair, teacher_origin_sub_obj_pair), "Student and teacher sub obj pair not equal!"

            projected_predicate_gated_visual_features = projected_predicate_gated_visual_features[negative_mask]
            projected_predicate_features = projected_predicate_features[negative_mask]
            projected_triplet_gated_visual_features = projected_triplet_gated_visual_features[negative_mask]
            projected_triplet_features = projected_triplet_features[negative_mask]
            predicate_combined_features_in_text_space = predicate_combined_features_in_text_space[negative_mask]
            triplet_combined_features_in_text_space = triplet_combined_features_in_text_space[negative_mask]

            teacher_projected_predicate_gated_visual_features = teacher_projected_predicate_gated_visual_features[negative_mask]
            teacher_projected_predicate_features = teacher_projected_predicate_features[negative_mask]
            teacher_projected_triplet_gated_visual_features = teacher_projected_triplet_gated_visual_features[negative_mask]
            teacher_projected_triplet_features = teacher_projected_triplet_features[negative_mask]
            teacher_predicate_combined_features_in_text_space = teacher_predicate_combined_features_in_text_space[negative_mask]
            teacher_triplet_combined_features_in_text_space = teacher_triplet_combined_features_in_text_space[negative_mask]

            distillation_loss = torch.tensor(0.0).to(predicate_gated_visual_loss.device)
            if negative_mask.numel() != 0 and negative_mask.any():
                distillation_loss += F.l1_loss(projected_predicate_gated_visual_features, teacher_projected_predicate_gated_visual_features, reduction='mean')
                distillation_loss += F.l1_loss(projected_predicate_features, teacher_projected_predicate_features, reduction='mean')
                distillation_loss += F.l1_loss(projected_triplet_gated_visual_features, teacher_projected_triplet_gated_visual_features, reduction='mean')
                distillation_loss += F.l1_loss(projected_triplet_features, teacher_projected_triplet_features, reduction='mean')
                distillation_loss += F.l1_loss(predicate_combined_features_in_text_space, teacher_predicate_combined_features_in_text_space, reduction='mean')
                distillation_loss += F.l1_loss(triplet_combined_features_in_text_space, teacher_triplet_combined_features_in_text_space, reduction='mean')


        losses['loss_predicate_contrastive'] = ((predicate_gated_visual_loss + predicate_loss) / 2.0)
        losses['loss_triplet_contrastive'] = ((triplet_gated_visual_loss + triplet_loss) / 2.0)
        if teacher_result_dict:
            losses['loss_distill'] = distillation_loss

        # losses['loss_predicate_contrastive'] = torch.tensor(0.0).to('cuda')
        # losses['loss_triplet_contrastive'] = torch.tensor(0.0).to('cuda')
        # if teacher_result_dict:
        #     losses['loss_distill'] = torch.tensor(0.0).to('cuda')

        # loss_dict = {
        #     'loss_predicate_contrastive': ((predicate_gated_visual_loss + predicate_loss) / 2.0),
        #     'loss_triplet_contrastive': ((triplet_gated_visual_loss + triplet_loss) / 2.0),
        #     'loss_predicate_classification': classification_loss
        # }

        return losses

    def get_flattened_result(self, result_dict, perm=None):
        '''
        each batch's pairs have been padded for attention layer, now conduct de-padding.
        '''
        pair_num_list = result_dict['pair_num_list']

        if not perm:
            projected_predicate_gated_visual_features = torch.cat(
                [result_dict['projected_predicate_gated_visual_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
                dim=0
                )
            projected_predicate_features = torch.cat(
                [result_dict['projected_predicate_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
                dim=0
                )
            projected_triplet_gated_visual_features = torch.cat(
                [result_dict['projected_triplet_gated_visual_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
                dim=0
                )
            projected_triplet_features = torch.cat(
                [result_dict['projected_triplet_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
                dim=0
                )
            predicate_combined_features_in_text_space = torch.cat(
                [result_dict['predicate_combined_features_in_text_space'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
                dim=0
                )
            triplet_combined_features_in_text_space = torch.cat(
                [result_dict['triplet_combined_features_in_text_space'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)],
                dim=0
                )
            triplet_labels = torch.cat(result_dict['triplet_labels'])
            origin_sub_obj_pair = torch.cat(result_dict['origin_sub_obj_pair'])
            return (
                projected_predicate_gated_visual_features,
                projected_predicate_features,
                projected_triplet_gated_visual_features,
                projected_triplet_features,
                predicate_combined_features_in_text_space,
                triplet_combined_features_in_text_space,
                triplet_labels,
                origin_sub_obj_pair
            )

        projected_predicate_gated_visual_features = [result_dict['projected_predicate_gated_visual_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)]
        projected_predicate_features = [result_dict['projected_predicate_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)]
        projected_triplet_gated_visual_features = [result_dict['projected_triplet_gated_visual_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)]
        projected_triplet_features = [result_dict['projected_triplet_features'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)]
        predicate_combined_features_in_text_space = [result_dict['predicate_combined_features_in_text_space'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)]
        triplet_combined_features_in_text_space = [result_dict['triplet_combined_features_in_text_space'][i, :pair_num] for i, pair_num in enumerate(pair_num_list)]
        triplet_labels = result_dict['triplet_labels']
        origin_sub_obj_pair = result_dict['origin_sub_obj_pair']

        for i, i_perm in enumerate(perm):
            projected_predicate_gated_visual_features[i] = projected_predicate_gated_visual_features[i][i_perm]
            projected_predicate_features[i] = projected_predicate_features[i][i_perm]
            projected_triplet_gated_visual_features[i] = projected_triplet_gated_visual_features[i][i_perm]
            projected_triplet_features[i] = projected_triplet_features[i][i_perm]
            predicate_combined_features_in_text_space[i] = predicate_combined_features_in_text_space[i][i_perm]
            triplet_combined_features_in_text_space[i] = triplet_combined_features_in_text_space[i][i_perm]
            triplet_labels[i] = triplet_labels[i][i_perm]
            origin_sub_obj_pair[i] = origin_sub_obj_pair[i][i_perm]
 
        projected_predicate_gated_visual_features = torch.cat(projected_predicate_gated_visual_features)
        projected_predicate_features = torch.cat(projected_predicate_features)
        projected_triplet_gated_visual_features = torch.cat(projected_triplet_gated_visual_features)
        projected_triplet_features = torch.cat(projected_triplet_features)
        predicate_combined_features_in_text_space = torch.cat(predicate_combined_features_in_text_space)
        triplet_combined_features_in_text_space = torch.cat(triplet_combined_features_in_text_space)
        triplet_labels = torch.cat(triplet_labels)
        origin_sub_obj_pair = torch.cat(origin_sub_obj_pair)

        return (
            projected_predicate_gated_visual_features,
            projected_predicate_features,
            projected_triplet_gated_visual_features,
            projected_triplet_features,
            predicate_combined_features_in_text_space,
            triplet_combined_features_in_text_space,
            triplet_labels,
            origin_sub_obj_pair
        )

    def i_contrastive(self, features, mask, valid_mask, device, tau):
        """
        Args:
            features: (num_pairs, d)
            mask: (valid, num_pairs), True for same category.
            valid_mask: (num_pairs,), True for valid samples.
        """
        if valid_mask.sum() == 0:
            # return torch.tensor(0.0, device=device, requires_grad=True)
            return features.sum() * 0.0

        source_features = features[valid_mask]  # (valid, d)
        destiny_features = features  # (num_pairs, d)

        logits_source_to_destiny = torch.div(torch.matmul(source_features, destiny_features.T), tau)  # (valid, num_pairs)
        logits_max, _ = torch.max(logits_source_to_destiny, dim=-1, keepdim=True)
        logits = logits_source_to_destiny - logits_max.detach()
        log_probability = torch.nn.functional.log_softmax(logits, dim=-1)  # (valid, num_pairs)
        loss = - torch.sum(mask * log_probability, dim=-1) / torch.sum(mask, dim=-1)
        return loss.mean()

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]


        self.global_iter = global_iter



        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {
                                  'rel_text_dict': outputs.get('rel_text_dict', None),
                                  'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                                 }
                        if outputs_t is not None:
                            kwargs['outputs_t'] = aux_outputs_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {
                              'rel_text_dict': outputs.get('rel_text_dict', None), 
                              'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                             }
                    if outputs_t is not None:
                        kwargs['outputs_t'] = interm_outputs_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups

    def generate_cut_slice_by_caption(self, captions):
        captions = [e[:-1].split('.') for e in captions]
        for caption in captions:
            caption = [('. '.join(e) + '. ') for e in caption]


class SetCriterionNoPostProcess(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)


        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes, rel_text_dict, rel_text_dict_t=None, outputs_t=None):
        """
            compute loss for relations 
        """
        batch_ids = outputs['batch_ids']
        relation_feature = outputs['relation_feature']
        all_edge_lbl = outputs['all_edge_lbl']
        freq_dist = outputs['freq_dist']

        relation_feature_t = None
        if outputs_t:
            relation_feature_t = outputs_t['relation_feature']

        relation_wo_labels = False
        relation_is_str = False
        rel_tgt = None

        # loss
        if self.rln_classifier is not None:
            assert not relation_wo_labels, "rln_classifier should be None for open vocabulary !"
            # rel_logits = self.rln_classifier(relation_feature)
            relation_feature = relation_feature.to(self.rln_classifier.module.weight.dtype if is_distributed() else self.rln_classifier.weight.dtype)
            rel_logits = self.rln_classifier(relation_feature)
            if self.rln_freq_bias is not None:
                # freq_dist = torch.cat(freq_dist, 0)[_idx_]
                rel_logits += freq_dist

            rel_tgt_onehot = torch.zeros([rel_logits.shape[0], rel_logits.shape[1]],
                                          dtype =rel_logits.dtype, 
                                          layout=rel_logits.layout, 
                                          device=rel_logits.device)

            all_edge_lbl = all_edge_lbl.to(rel_logits.device)
            rel_tgt_onehot.scatter_(-1, all_edge_lbl.unsqueeze(-1), 1)
        else:
            encoded_text = rel_text_dict['encoded_text'][batch_ids]
            text_mask = rel_text_dict['text_token_mask'][batch_ids]
            input_ids = rel_text_dict['input_ids'][batch_ids]

            rel_logits  = torch.einsum("a d, a b d -> a b", relation_feature, encoded_text)
            rel_logits.masked_fill_(~text_mask, float('-inf'))
            # padding to max_text_len
            rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
            rel_tgt_onehot = torch.zeros_like(rel_logits)

            if relation_feature_t is not None and not self.unsupervised_distill:
                with torch.no_grad():
                    encoded_text_t = rel_text_dict_t['encoded_text'][batch_ids]
                    rel_logits_t  = torch.einsum("a d, a b d -> a b", relation_feature_t, encoded_text_t)
                    rel_logits_t.masked_fill_(~text_mask, float('-inf'))
                    rel_logits_t = padding_last(rel_logits_t, rel_logits.shape[-1])

                rel_logits_t = shrink_sigmoid(rel_logits_t, 2.0) # suitable for OvR
                #rel_logits_t.sigmoid_()

            if relation_wo_labels or relation_is_str:
                for ii, name in enumerate(rel_tgt):
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name + '.').input_ids[1:-1]
                    all_ids = input_ids[ii]

                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids: {}".format(
                            name, self.tokenizer.decode(all_ids))
                    assert start_i < rel_tgt_onehot.shape[1] and end_i < rel_tgt_onehot.shape[1], "start_i:{}, end_i:{}, tgt shape:{}".format(start_i, end_i, rel_tgt_onehot.shape)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0
            else:
                for ii, label in enumerate(all_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]' and relation_feature_t is not None and not self.unsupervised_distill: # use teacher's output
                        rel_tgt_onehot[ii] = rel_logits_t[ii]
                        continue 

                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    all_ids = input_ids[ii]
                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0

        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

                
        if not self.focal_loss_for_edges:  # CE
            loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
            losses = dict(loss_edges=loss)
        else: # focal loss
            alpha, gamma = 0.25, 2.0
            eps = 1e-5 
            rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
            rel_mask = (rel_logits != float('-inf')).float()
          
            rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
            rel_weight[rel_weight == 0] = 1.0

            pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
            neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

            pos_loss = pos_loss.sum() 
            neg_loss = neg_loss.sum() 


            loss = (pos_loss + neg_loss) / rel_num

            losses = dict(loss_edges=loss)
            with torch.no_grad():
                losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
                losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

            if outputs_t is not None and self.unsupervised_distill:
                ids = torch.where(all_edge_lbl == 0)[0]
                loss_distill = F.l1_loss(relation_feature[ids], relation_feature_t[ids], reduction='sum') / rel_num

                losses['loss_distill'] = loss_distill


        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)
        if os.environ.get("DEBUG") == '1':
            import pdb; pdb.set_trace()

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]


        self.global_iter = global_iter



        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device


        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {
                                  'rel_text_dict': outputs.get('rel_text_dict', None),
                                  'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                                 }
                        if outputs_t is not None:
                            kwargs['outputs_t'] = aux_outputs_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {
                              'rel_text_dict': outputs.get('rel_text_dict', None), 
                              'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                             }
                    if outputs_t is not None:
                        kwargs['outputs_t'] = interm_outputs_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class SetCriterionDecoupled(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)


        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None)  and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes, rel_text_dict, rel_text_dict_t=None, outputs_t=None):
        """
            compute loss for relations 
        """
        def get_layer_loss(batch_ids, relation_feature, all_edge_lbl, freq_dist, rel_text_dict, rel_text_dict_t=None, outputs_t=None):
            relation_feature_t = None
            if outputs_t:
                relation_feature_t = outputs_t['relation_feature']

            relation_wo_labels = False
            relation_is_str = False
            rel_tgt = None

            # loss
            if self.rln_classifier is not None:
                assert not relation_wo_labels, "rln_classifier should be None for open vocabulary !"
                # rel_logits = self.rln_classifier(relation_feature)
                relation_feature = relation_feature.to(self.rln_classifier.module.weight.dtype if is_distributed() else self.rln_classifier.weight.dtype)
                rel_logits = self.rln_classifier(relation_feature)
                if self.rln_freq_bias is not None:
                    # freq_dist = torch.cat(freq_dist, 0)[_idx_]
                    rel_logits += freq_dist

                rel_tgt_onehot = torch.zeros([rel_logits.shape[0], rel_logits.shape[1]],
                                            dtype =rel_logits.dtype, 
                                            layout=rel_logits.layout, 
                                            device=rel_logits.device)

                all_edge_lbl = all_edge_lbl.to(rel_logits.device)
                rel_tgt_onehot.scatter_(-1, all_edge_lbl.unsqueeze(-1), 1)
            else:
                encoded_text = rel_text_dict['encoded_text'][batch_ids]
                text_mask = rel_text_dict['text_token_mask'][batch_ids]
                input_ids = rel_text_dict['input_ids'][batch_ids]

                rel_logits  = torch.einsum("a d, a b d -> a b", relation_feature, encoded_text)
                rel_logits.masked_fill_(~text_mask, float('-inf'))
                # padding to max_text_len
                rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
                rel_tgt_onehot = torch.zeros_like(rel_logits)  # (num_relations, text_length)

                if relation_feature_t is not None and not self.unsupervised_distill:
                    with torch.no_grad():
                        encoded_text_t = rel_text_dict_t['encoded_text'][batch_ids]
                        rel_logits_t  = torch.einsum("a d, a b d -> a b", relation_feature_t, encoded_text_t)
                        rel_logits_t.masked_fill_(~text_mask, float('-inf'))
                        rel_logits_t = padding_last(rel_logits_t, rel_logits.shape[-1])

                    rel_logits_t = shrink_sigmoid(rel_logits_t, 2.0) # suitable for OvR
                    #rel_logits_t.sigmoid_()

                if relation_wo_labels or relation_is_str:
                    for ii, name in enumerate(rel_tgt):
                        if name == '[UNK]':
                            continue 

                        ids = self.tokenizer(name + '.').input_ids[1:-1]
                        all_ids = input_ids[ii]

                        start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                        assert start_i != end_i, "cannot find query:{} from input_ids: {}".format(
                                name, self.tokenizer.decode(all_ids))
                        assert start_i < rel_tgt_onehot.shape[1] and end_i < rel_tgt_onehot.shape[1], "start_i:{}, end_i:{}, tgt shape:{}".format(start_i, end_i, rel_tgt_onehot.shape)

                        rel_tgt_onehot[ii, start_i: end_i] = 1.0
                else:
                    for ii, label in enumerate(all_edge_lbl.tolist()):  # (num_rel_feats,)
                        name = self.ind_to_predicates[label]
                        if name == '[UNK]' and relation_feature_t is not None and not self.unsupervised_distill: # use teacher's output
                            rel_tgt_onehot[ii] = rel_logits_t[ii]
                            continue 

                        if name == '[UNK]':
                            continue 

                        ids = self.tokenizer(name +'.').input_ids[1:-1]
                        all_ids = input_ids[ii]  # (num_rel_feats, text_length)
                        start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                        assert start_i < end_i, "cannot find query:{} from input_ids".format(ids)

                        rel_tgt_onehot[ii, start_i: end_i] = 1.0

            rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
            #rel_num = rel_tgt_onehot.sum()
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(rel_num)
            rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()
                    
            if not self.focal_loss_for_edges:  # CE
                loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
                losses = dict(loss_edges=loss)
            else: # focal loss
                alpha, gamma = 0.25, 2.0
                eps = 1e-5 
                rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)
                rel_mask = (rel_logits != float('-inf')).float()
            
                rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
                rel_weight[rel_weight == 0] = 1.0

                pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
                neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

                pos_loss = pos_loss.sum() 
                neg_loss = neg_loss.sum() 


                loss = (pos_loss + neg_loss) / rel_num

                losses = dict(loss_edges=loss)
                with torch.no_grad():
                    losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
                    losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

                if outputs_t is not None and self.unsupervised_distill:
                    ids = torch.where(all_edge_lbl == 0)[0]
                    loss_distill = F.l1_loss(relation_feature[ids], relation_feature_t[ids], reduction='sum') / rel_num

                    losses['loss_distill'] = loss_distill

            losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)
            if os.environ.get("DEBUG") == '1':
                import pdb; pdb.set_trace()

            return losses

        batch_ids = outputs['batch_ids']
        relation_feature = outputs['relation_feature']
        all_edge_lbl = outputs['all_edge_lbl']
        freq_dist = outputs['freq_dist']
        layer_num = len(relation_feature)

        layer_loss = []
        for i in range(layer_num):
            layer_loss.append(get_layer_loss(batch_ids, relation_feature[i], all_edge_lbl, freq_dist, rel_text_dict, rel_text_dict_t, outputs_t))

        loss_dict = {}
        for i in range(layer_num):
            if i < layer_num - 1:
                loss_dict[f'loss_edges_{i}'] = layer_loss[i]['loss_edges']
                loss_dict[f'loss_edges_pos_{i}'] = layer_loss[i]['loss_edges_pos']
                loss_dict[f'loss_edges_neg_{i}'] = layer_loss[i]['loss_edges_neg']
            else:
                loss_dict[f'loss_edges'] = layer_loss[i]['loss_edges']
                loss_dict[f'loss_edges_pos'] = layer_loss[i]['loss_edges_pos']
                loss_dict[f'loss_edges_neg'] = layer_loss[i]['loss_edges_neg']

        return loss_dict

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]

        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None 
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        continue

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    continue

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


# class OvSGTRBaseCriterion(nn.Module):
#     """ This class computes the loss for Conditional DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """
#     def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
#                 **kwargs):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of object categories, omitting the special no-object category
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#             focal_alpha: alpha in Focal Loss
#         """
#         super().__init__()
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
#         self.focal_alpha = focal_alpha
#         self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

#         self.tokenizer = kwargs.get("tokenizer", None)
#         self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
#         self.global_iter = -1
#         hidden_dim = kwargs.get("hidden_dim", 256)
#         self.min_obj=-hidden_dim*math.log(0.9)
#         self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
#         self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
#         self.obj_threshold = kwargs.get("obj_threshold", 0.5)

#         self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
#         self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

#         self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (Binary focal loss)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         idx = self._get_src_permutation_idx(indices)

#         src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
#         src_mask = src_logits == float('-inf')
#         # 
#         tgt_pos_seg = []
#         for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
#             gt_names = target['gt_names']  # a list of gt box's name
#             all_ids = target['input_ids']
#             pos_seg = []
#             for name in gt_names:
#                 ids = self.tokenizer(name +'.').input_ids[1:-1]
#                 start_i, end_i = search_query_pos(all_ids.tolist(), ids)
#                 assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
#                                 self.tokenizer.decode(all_ids))
#                 pos_seg.append((start_i, end_i))

#             for j in J.tolist():
#                 tgt_pos_seg.append(pos_seg[j])


#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


#         # set positive labels
#         for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
#             target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

#         target_classes_onehot = target_classes_onehot[:,:,:-1]

#         alpha = self.focal_alpha
#         gamma = 2.0

#         loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
#                                      alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
#         loss_ce.masked_fill_(src_mask, 0.)

#         nb_pos = target_classes_onehot.sum(-1)
#         nb_pos[nb_pos == 0] = 1.0

#         loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
#         losses = {'loss_ce': loss_ce}


#         return losses

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes

#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))

#         losses['loss_giou'] = loss_giou.sum() / num_boxes

#         ## calculate the x,y and h,w loss
#         #with torch.no_grad():
#         #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
#         #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

#         return losses

#     def loss_masks(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs

#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)
#         src_masks = outputs["pred_masks"]
#         src_masks = src_masks[src_idx]
#         masks = [t["masks"] for t in targets]
#         # TODO use valid to mask invalid areas due to padding in loss
#         target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
#         target_masks = target_masks.to(src_masks)
#         target_masks = target_masks[tgt_idx]

#         # upsample predictions to the target size
#         src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
#                                 mode="bilinear", align_corners=False)
#         src_masks = src_masks[:, 0].flatten(1)

#         target_masks = target_masks.flatten(1)
#         target_masks = target_masks.view(src_masks.shape)
#         losses = {
#             "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
#             "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
#         }
#         return losses

#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'labels': self.loss_labels,
#             'boxes': self.loss_boxes,
#             'masks': self.loss_masks,
#             'edges': self.loss_edges,
#             'obj_likelihood': self.loss_obj_likelihood,
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
#         assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

#         idx = self._get_src_permutation_idx(indices)
#         pred_obj = outputs["pred_obj"][idx]
#         return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

#     @torch.no_grad()
#     def get_objectness(self, pred_obj):
#         return torch.exp(-self.obj_temp*pred_obj)

#     def loss_edges(self, outputs, targets, indices, num_boxes, relation_features, edge_lbls, rel_text_dict):
#         """
#           compute loss for relations 
#         """
#         # Flatten outputs for loss calculation.
#         batch_pair_num = [len(e) for e in relation_features]
#         batch_pair_num = torch.tensor(batch_pair_num, device=relation_features[0].device)
#         encoded_text = torch.repeat_interleave(rel_text_dict['encoded_text'], batch_pair_num, dim=0)
#         text_mask = torch.repeat_interleave(rel_text_dict['text_token_mask'], batch_pair_num, dim=0)
#         input_ids = torch.repeat_interleave(rel_text_dict['input_ids'], batch_pair_num, dim=0)

#         relation_features = torch.cat(relation_features)
#         all_edge_lbl = torch.cat(edge_lbls)

#         rel_logits = torch.einsum("a d, a b d -> a b", relation_features, encoded_text)
#         rel_logits.masked_fill_(~text_mask, float('-inf'))
#         # padding to max_text_len
#         rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
#         rel_tgt_onehot = torch.zeros_like(rel_logits)

#         for ii, label in enumerate(all_edge_lbl.tolist()):
#             name = self.ind_to_predicates[label]
#             if name == '[UNK]':
#                 continue 

#             ids = self.tokenizer(name +'.').input_ids[1:-1]
#             all_ids = input_ids[ii]
#             start_i, end_i = search_query_pos(all_ids.tolist(), ids)
#             assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

#             rel_tgt_onehot[ii, start_i: end_i] = 1.0


#         rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
#         #rel_num = rel_tgt_onehot.sum()
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(rel_num)
#         rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()


#         if not self.focal_loss_for_edges:  # CE
#             loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
#             losses = dict(loss_edges=loss)
#         else: # focal loss
#             alpha, gamma = 0.25, 2.0
#             eps = 1e-5 

#             rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
#             rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)


#             rel_mask = (rel_logits != float('-inf')).float()
          
#             rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
#             rel_weight[rel_weight == 0] = 1.0

#             pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
#             neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

#             pos_loss = pos_loss.sum() 
#             neg_loss = neg_loss.sum() 


#             loss = (pos_loss + neg_loss) / rel_num

#             losses = dict(loss_edges=loss)
#             with torch.no_grad():
#                 losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
#                 losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

#         losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)

#         return losses

#     def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
#         prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
#         mask = (logits != float('-inf')).float()

#         pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
#         neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

#         num_pos = tgt_onehot.sum()
#         if num_pos == 0:
#             return neg_loss.sum()

#         loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

#         return loss

#     def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
#                 global_iter=-1):

#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
            
#              return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

#         """
#         input_ids = outputs['input_ids']
#         for bid, target in enumerate(targets):
#             target['input_ids'] = input_ids[bid]

#         self.global_iter = global_iter

#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
#         device=next(iter(outputs.values())).device

#         # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
#         indices = self.matcher(outputs_without_aux, targets)
        
#         if outputs_t is not None:
#             outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
#             indices_t = self.matcher(outputs_without_aux_t, targets)
#         else:
#             indices_t = None

#         if return_indices:
#             indices0_copy = indices
#             indices_list = []

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["boxes"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}

#         # prepare for dn loss
#         dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

#         if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
#             output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

#             dn_pos_idx = []
#             dn_neg_idx = []
#             for i in range(len(targets)):
#                 if len(targets[i]['boxes']) > 0:
#                     t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
#                     t = t.unsqueeze(0).repeat(scalar, 1)
#                     tgt_idx = t.flatten()
#                     output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
#                     output_idx = output_idx.flatten()
#                 else:
#                     output_idx = tgt_idx = torch.tensor([]).long().cuda()

#                 dn_pos_idx.append((output_idx, tgt_idx))
#                 dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

#             output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
#             l_dict = {}
#             for loss in self.losses:
#                 if 'edges' == loss:
#                     continue 
#                 if 'obj_likelihood' == loss:
#                     continue 

#                 kwargs = {}
#                 if 'labels' in loss:
#                     kwargs = {'log': False}
#                 l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

#             l_dict = {k + f'_dn': v for k, v in l_dict.items()}
#             losses.update(l_dict)
#         else:
#             pass

#         for loss in self.losses:
#             kwargs = {}
#             if 'edges' in loss:
#                 kwargs = {'relation_features': outputs['relation_features'],
#                           'edge_lbls': outputs['edge_lbls'],
#                           'rel_text_dict': outputs['rel_text_dict'],
#                           }
#                 if outputs_t is not None:
#                     kwargs['outputs_t'] = outputs_t
#                     kwargs['indices_t'] = indices_t

#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for idx, aux_outputs in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_outputs, targets)
#                 if outputs_t is not None:
#                     aux_outputs_t = outputs_t['aux_outputs'][idx]
#                     indices_t = self.matcher(aux_outputs_t, targets)

#                 if return_indices:
#                     indices_list.append(indices)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}
#                     if 'edges' in loss:
#                         kwargs = {'relation_features': outputs['relation_features'],
#                                 'edge_lbls': outputs['edge_lbls'],
#                                 'rel_text_dict': outputs['rel_text_dict'],
#                                 }
#                         if outputs_t is not None:
#                             kwargs['outputs_t'] = aux_outputs_t
#                             kwargs['indices_t'] = indices_t

#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#                 if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
#                     aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
#                     l_dict={}
#                     for loss in self.losses:
#                         if 'edges' == loss:
#                             continue 

#                         kwargs = {}
#                         if 'labels' in loss:
#                             kwargs = {'log': False}

#                         l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
#                                                                  **kwargs))

#                     l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#                 else:
#                     pass

#         # interm_outputs loss
#         if 'interm_outputs' in outputs:
#             interm_outputs = outputs['interm_outputs']
#             indices = self.matcher(interm_outputs, targets)

#             if outputs_t is not None:
#                 interm_outputs_t = outputs_t['interm_outputs']
#                 indices_t = self.matcher(interm_outputs_t, targets)

#             if return_indices:
#                 indices_list.append(indices)
#             for loss in self.losses:
#                 if loss == 'masks':
#                     # Intermediate masks losses are too costly to compute, we ignore them.
#                     continue
#                 if 'obj_likelihood' == loss:
#                     continue 
#                 kwargs = {}
#                 if loss == 'labels':
#                     # Logging is enabled only for the last layer
#                     kwargs = {'log': False}

#                 if 'edges' in loss:
#                     kwargs = {'relation_features': outputs['relation_features'],
#                             'edge_lbls': outputs['edge_lbls'],
#                             'rel_text_dict': outputs['rel_text_dict'],
#                             }
#                     if outputs_t is not None:
#                         kwargs['outputs_t'] = interm_outputs_t
#                         kwargs['indices_t'] = indices_t

#                 l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
#                 l_dict = {k + f'_interm': v for k, v in l_dict.items()}
#                 losses.update(l_dict)

#         # enc output loss
#         if 'enc_outputs' in outputs:
#             for i, enc_outputs in enumerate(outputs['enc_outputs']):
#                 indices = self.matcher(enc_outputs, targets)
#                 if return_indices:
#                     indices_list.append(indices)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     if 'obj_likelihood' == loss:
#                         continue 
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}
#                     if 'edges' == loss:
#                         continue 

#                     l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         if return_indices:
#             indices_list.append(indices0_copy)
#             return losses, indices_list

#         return losses

#     def prep_for_dn(self,dn_meta):
#         output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
#         num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
#         assert pad_size % num_dn_groups==0
#         single_pad=pad_size//num_dn_groups

#         return output_known_lbs_bboxes,single_pad,num_dn_groups


class OvSGTRBaseCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes, relation_features, edge_lbls, rel_text_dict, relation_features_t=None, edge_lbls_t=None):
        """
        compute loss for relations 
        """
        # Flatten outputs for loss calculation.
        batch_pair_num = [len(e) for e in relation_features]
        batch_pair_num = torch.tensor(batch_pair_num, device=relation_features[0].device)
        encoded_text = torch.repeat_interleave(rel_text_dict['encoded_text'], batch_pair_num, dim=0)
        text_mask = torch.repeat_interleave(rel_text_dict['text_token_mask'], batch_pair_num, dim=0)
        input_ids = torch.repeat_interleave(rel_text_dict['input_ids'], batch_pair_num, dim=0)

        relation_features = torch.cat(relation_features)
        all_edge_lbl = torch.cat(edge_lbls)

        if edge_lbls_t is not None:
            relation_features_t = torch.cat(relation_features_t)
            all_edge_lbls_t = torch.cat(edge_lbls_t)
            assert torch.equal(all_edge_lbls_t, all_edge_lbl)

        rel_logits = torch.einsum("a d, a b d -> a b", relation_features, encoded_text)
        rel_logits.masked_fill_(~text_mask, float('-inf'))
        # padding to max_text_len
        rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
        rel_tgt_onehot = torch.zeros_like(rel_logits)

        for ii, label in enumerate(all_edge_lbl.tolist()):
            name = self.ind_to_predicates[label]
            if name == '[UNK]':
                continue 

            ids = self.tokenizer(name +'.').input_ids[1:-1]
            all_ids = input_ids[ii]
            start_i, end_i = search_query_pos(all_ids.tolist(), ids)
            assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

            rel_tgt_onehot[ii, start_i: end_i] = 1.0


        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()


        if not self.focal_loss_for_edges:  # CE
            loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
            losses = dict(loss_edges=loss)
        else: # focal loss
            alpha, gamma = 0.25, 2.0
            eps = 1e-5 

            rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
            rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)


            rel_mask = (rel_logits != float('-inf')).float()
          
            rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
            rel_weight[rel_weight == 0] = 1.0

            pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
            neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

            pos_loss = pos_loss.sum() 
            neg_loss = neg_loss.sum() 


            loss = (pos_loss + neg_loss) / rel_num

            losses = dict(loss_edges=loss)
            with torch.no_grad():
                losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
                losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

        if relation_features_t is not None:
            ids = torch.where(all_edge_lbl == 0)[0].long()
            loss_distill = F.l1_loss(relation_features[ids], relation_features_t[ids], reduction='sum') / rel_num
            losses['loss_distill'] = loss_distill

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]

        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue 
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {'relation_features': outputs['relation_features'],
                          'edge_lbls': outputs['edge_lbls'],
                          'rel_text_dict': outputs['rel_text_dict'],
                          }
                if outputs_t is not None:
                    kwargs['relation_features_t'] = outputs_t['relation_features']
                    kwargs['edge_lbls_t'] = outputs_t['edge_lbls']
                    # kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {'relation_features': outputs['relation_features'],
                                'edge_lbls': outputs['edge_lbls'],
                                'rel_text_dict': outputs['rel_text_dict'],
                                }
                        if outputs_t is not None:
                            kwargs['relation_features_t'] = outputs_t['relation_features']
                            kwargs['edge_lbls_t'] = outputs_t['edge_lbls']
                            # edge_lbls_t
                            # kwargs['outputs_t'] = aux_outputs_t
                            # kwargs['indices_t'] = indices_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {'relation_features': outputs['relation_features'],
                            'edge_lbls': outputs['edge_lbls'],
                            'rel_text_dict': outputs['rel_text_dict'],
                            }
                    if outputs_t is not None:
                        kwargs['relation_features_t'] = outputs_t['relation_features']
                        kwargs['edge_lbls_t'] = outputs_t['edge_lbls']
                        # kwargs['outputs_t'] = interm_outputs_t
                        # kwargs['indices_t'] = indices_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups



class OvSGTRTextCrossCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes, relation_features, edge_lbls, rel_text_dict):
        """
          compute loss for relations 
        """
        # Flatten outputs for loss calculation.
        batch_pair_num = [len(e) for e in relation_features]
        batch_pair_num = torch.tensor(batch_pair_num, device=relation_features[0].device)
        encoded_text = torch.repeat_interleave(rel_text_dict['encoded_text'], batch_pair_num, dim=0)
        text_mask = torch.repeat_interleave(rel_text_dict['text_token_mask'], batch_pair_num, dim=0)
        input_ids = torch.repeat_interleave(rel_text_dict['input_ids'], batch_pair_num, dim=0)

        relation_features = torch.cat(relation_features)
        all_edge_lbl = torch.cat(edge_lbls)

        rel_logits = torch.einsum("a d, a b d -> a b", relation_features, encoded_text)
        rel_logits.masked_fill_(~text_mask, float('-inf'))
        # padding to max_text_len
        rel_logits = padding_last(rel_logits, 512) # 512 ? 2048
        rel_tgt_onehot = torch.zeros_like(rel_logits)

        for ii, label in enumerate(all_edge_lbl.tolist()):
            name = self.ind_to_predicates[label]
            if name == '[UNK]':
                continue 

            ids = self.tokenizer(name +'.').input_ids[1:-1]
            all_ids = input_ids[ii]
            start_i, end_i = search_query_pos(all_ids.tolist(), ids)
            assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

            rel_tgt_onehot[ii, start_i: end_i] = 1.0


        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()


        if not self.focal_loss_for_edges:  # CE
            loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
            losses = dict(loss_edges=loss)
        else: # focal loss
            alpha, gamma = 0.25, 2.0
            eps = 1e-5 

            rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
            rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)


            rel_mask = (rel_logits != float('-inf')).float()
          
            rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
            rel_weight[rel_weight == 0] = 1.0

            pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
            neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

            pos_loss = pos_loss.sum() 
            neg_loss = neg_loss.sum() 


            loss = (pos_loss + neg_loss) / rel_num

            losses = dict(loss_edges=loss)
            with torch.no_grad():
                losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
                losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]

        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {'relation_features': outputs['relation_features'],
                          'edge_lbls': outputs['edge_lbls'],
                          'rel_text_dict': outputs['rel_text_dict'],
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t
                    kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        continue

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    continue

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class OvSGTRVS3LikeCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes, rln_pred_logits, edge_lbls, rel_text_dict):
        """
          compute loss for relations 
        """
        # Flatten outputs for loss calculation.
        rel_logits = rln_pred_logits
        # pad_size = 512 - rel_logits.size(-1)
        # if pad_size > 0:
        #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

        for i, i_rel_logits in enumerate(rel_logits):
            pad_size = 512 - i_rel_logits.size(-1)
            if pad_size > 0:
                i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
            rel_logits[i] = i_rel_logits

        rel_tgt_onehot = []
        for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
            i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

            if self.sgg_mode == 'full':
                i_rel_tgt_onehot[i_edge_lbl] = 1.0
            else:
                for ii, label in enumerate(i_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        rel_logits = torch.cat(rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)
        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 

        rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
        rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

        rel_mask = (rel_logits != float('-inf')).float()
        
        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum() 
        neg_loss = neg_loss.sum() 

        loss = (pos_loss + neg_loss) / rel_num

        losses = dict(loss_edges=loss)
        with torch.no_grad():
            losses['loss_edges_pos'] = pos_loss.detach() /  rel_num
            losses['loss_edges_neg'] = neg_loss.detach() /  rel_num

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]

        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue
                if 'obj_likelihood' == loss:
                    continue 

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {
                    'rln_pred_logits': outputs['rln_pred_logits'],
                    'edge_lbls': outputs['edge_lbls'],
                    'rel_text_dict': outputs['rel_text_dict'],
                    }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t
                    kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        continue

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue 

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    continue

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue 

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class OvSGTRRelDecoderCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
        """
        rln_pred_boxes: (bs, n, 4)
        rln_gt_boxes:   (bs, n, 4)
        rln_mask: (bs, n)  True = padding / invalid
        """
        # flatten
        src_boxes = rln_pred_boxes.reshape(-1, 4)
        target_boxes = rln_gt_boxes.reshape(-1, 4)
        valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

        # keep only the valid boxes
        src_boxes = src_boxes[~valid_mask_flat]
        target_boxes = target_boxes[~valid_mask_flat]

        # guard against an all-padding batch (edge case)
        if src_boxes.numel() == 0:
            return {
                'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
                'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
            }

        # ---------------- L1 ----------------
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_rln_boxes

        # ---------------- GIoU ----------------
        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

        giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.sum() / num_rln_boxes

        return {
            'loss_rln_bbox': loss_bbox,
            'loss_rln_giou': loss_giou,
        }

    def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
        """
          compute loss for relations 
        """
        # Flatten outputs for loss calculation.
        rel_logits = [e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)]
        # pad_size = 512 - rel_logits.size(-1)
        # if pad_size > 0:
        #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

        # for i, i_rel_logits in enumerate(rel_logits):
        #     pad_size = 512 - i_rel_logits.size(-1)
        #     if pad_size > 0:
        #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
        #     rel_logits[i] = i_rel_logits

        rel_tgt_onehot = []
        for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
            i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

            if self.sgg_mode == 'full':
                i_rel_tgt_onehot[i_edge_lbl] = 1.0
            else:
                for ii, label in enumerate(i_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        rel_logits = torch.cat(rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)
        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 

        rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
        rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

        rel_mask = (rel_logits != float('-inf')).float()
        
        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum() 
        neg_loss = neg_loss.sum() 

        loss = (pos_loss + neg_loss) / rel_num

        losses = dict(loss_rln_labels=loss)
        with torch.no_grad():
            losses['loss_rln_labels_pos'] = pos_loss.detach() /  rel_num
            losses['loss_rln_labels_neg'] = neg_loss.detach() /  rel_num

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        if 'input_ids' not in targets[0].keys():
            input_ids = outputs['text_dict']['input_ids']
            for bid, target in enumerate(targets):
                target['input_ids'] = input_ids[bid]
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # =============== dn loss ===============
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        # =============== standard loss ===============
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # =============== aux loss ===============
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_output in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_output, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # =============== interm loss ===============
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # =============== enc_outputs loss ===============
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # =============== rln_decoder loss ===============
        rln_out = outputs['rln_out']
        rln_text_dict = outputs['rel_text_dict']
        rln_query_mask = rln_out['rln_query_mask']
        batch_pair_num = rln_out['batch_pair_num']
        edge_gt_lbls = outputs['edge_gt_lbls']
        edge_gt_boxes = outputs['edge_gt_boxes']

        rln_mask = (edge_gt_lbls == 0)
        edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls, batch_pair_num)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_rln_boxes = sum(len(t["edges"]) for t in targets)
        num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_rln_boxes)
        num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

        # interm class/box
        interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
        interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
        losses.update(interm_loss_rln_boxes_dict)
        losses.update(interm_loss_rln_labels_dict)

        # aux class/box
        for idx, aux_output in enumerate(rln_out['aux_outputs']):
            aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
            aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
            aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
            aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
            losses.update(aux_loss_rln_boxes_dict)
            losses.update(aux_loss_rln_labels_dict)

        # standard class / box
        loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        losses.update(loss_rln_boxes_dict)
        losses.update(loss_rln_labels_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class OvSGTRRelDecoderPairPruningCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
        """
        rln_pred_boxes: (bs, n, 4)
        rln_gt_boxes:   (bs, n, 4)
        rln_mask: (bs, n)  True = padding / invalid
        """
        # flatten
        src_boxes = rln_pred_boxes.reshape(-1, 4)
        target_boxes = rln_gt_boxes.reshape(-1, 4)
        valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

        # keep only the valid boxes
        src_boxes = src_boxes[~valid_mask_flat]
        target_boxes = target_boxes[~valid_mask_flat]

        # guard against an all-padding batch (edge case)
        if src_boxes.numel() == 0:
            # return {
            #     'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
            #     'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
            # }
            dummy_loss = rln_pred_boxes.sum() * 0.0
            return {
                'loss_rln_bbox': dummy_loss,
                'loss_rln_giou': dummy_loss,
            }

        # ---------------- L1 ----------------
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_rln_boxes

        # ---------------- GIoU ----------------
        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

        giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.sum() / num_rln_boxes

        return {
            'loss_rln_bbox': loss_bbox,
            'loss_rln_giou': loss_giou,
        }

    def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
        """
          compute loss for relations 
        """
        # Flatten outputs for loss calculation.
        rel_logits = [e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)]
        # pad_size = 512 - rel_logits.size(-1)
        # if pad_size > 0:
        #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

        # for i, i_rel_logits in enumerate(rel_logits):
        #     pad_size = 512 - i_rel_logits.size(-1)
        #     if pad_size > 0:
        #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
        #     rel_logits[i] = i_rel_logits

        rel_tgt_onehot = []
        for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
            i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

            if self.sgg_mode == 'full':
                i_rel_tgt_onehot[i_edge_lbl] = 1.0
            else:
                for ii, label in enumerate(i_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        rel_logits = torch.cat(rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)
        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 

        rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
        rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

        rel_mask = (rel_logits != float('-inf')).float()
        
        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum() 
        neg_loss = neg_loss.sum() 

        loss = (pos_loss + neg_loss) / rel_num

        losses = dict(loss_rln_labels=loss)
        with torch.no_grad():
            losses['loss_rln_labels_pos'] = pos_loss.detach() /  rel_num
            losses['loss_rln_labels_neg'] = neg_loss.detach() /  rel_num

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

        return losses

    def loss_pair_pruning(self, rln_pred_logits, rln_exist_score, batch_pair_num):
        """
        rln_pred_logits: (b, n, text_length)
        rln_exist_score: (b, n)   # logits
        batch_pair_num: list[int]
        """

        b, n, _ = rln_pred_logits.shape
        device = rln_pred_logits.device

        # ---- 1. relation prob ----
        prob = F.softmax(rln_pred_logits, dim=-1)   # (b, n, text_length)

        # ---- 2. max predicate prob ----
        r = prob.max(dim=-1).values   # (b, n)

        # ---- 3. detach (VERY IMPORTANT) ----
        r = r.detach()

        # ---- 4. sharpening, reduces noise----
        # tau = 0.7
        # r = r ** (1.0 / tau)

        # ---- 5. build valid mask ----
        mask = torch.zeros((b, n), device=device)

        for i in range(b):
            mask[i, :batch_pair_num[i]] = 1.0

        # ---- 6. optional: confidence mask (recommended)----
        # conf_mask = (r > 0.2).float()
        # mask = mask * conf_mask

        # ---- 7. BCE loss ----
        loss = F.binary_cross_entropy_with_logits(rln_exist_score, r, reduction='none')

        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        loss_dict = {"loss_pair_pruning": loss}

        return loss_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        if 'input_ids' not in targets[0].keys():
            input_ids = outputs['text_dict']['input_ids']
            for bid, target in enumerate(targets):
                target['input_ids'] = input_ids[bid]
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # =============== dn loss ===============
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        # =============== standard loss ===============
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # =============== aux loss ===============
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_output in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_output, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # =============== interm loss ===============
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # =============== enc_outputs loss ===============
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # =============== rln_decoder loss ===============
        rln_out = outputs['rln_out']
        rln_text_dict = outputs['rel_text_dict']
        rln_query_mask = rln_out['rln_query_mask']
        batch_pair_num = rln_out['batch_pair_num']
        edge_gt_lbls = outputs['edge_gt_lbls']
        edge_gt_boxes = outputs['edge_gt_boxes']

        rln_mask = (edge_gt_lbls == 0)
        edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls, batch_pair_num)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_rln_boxes = sum(len(t["edges"]) for t in targets)
        num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_rln_boxes)
        num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

        # pair pruning loss
        pair_pruning_loss = self.loss_pair_pruning(rln_out['rln_pred_logits'], rln_out['rln_exist_score'], batch_pair_num)
        losses.update(pair_pruning_loss)

        # interm class/box
        interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
        interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
        losses.update(interm_loss_rln_boxes_dict)
        losses.update(interm_loss_rln_labels_dict)

        # aux class/box
        for idx, aux_output in enumerate(rln_out['aux_outputs']):
            aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
            aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
            aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
            aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
            losses.update(aux_loss_rln_boxes_dict)
            losses.update(aux_loss_rln_labels_dict)

        # standard class / box
        loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        losses.update(loss_rln_boxes_dict)
        losses.update(loss_rln_labels_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


# class OvSGTRRelDecoderPairPruningCriterionV2(nn.Module):
#     """ This class computes the loss for Conditional DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """
#     def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
#                 **kwargs):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of object categories, omitting the special no-object category
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#             focal_alpha: alpha in Focal Loss
#         """
#         super().__init__()
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
#         self.focal_alpha = focal_alpha
#         self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

#         self.tokenizer = kwargs.get("tokenizer", None)
#         self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
#         self.global_iter = -1
#         hidden_dim = kwargs.get("hidden_dim", 256)
#         self.min_obj=-hidden_dim*math.log(0.9)
#         self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
#         self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
#         self.obj_threshold = kwargs.get("obj_threshold", 0.5)

#         self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
#         self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

#         self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (Binary focal loss)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         idx = self._get_src_permutation_idx(indices)

#         src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
#         src_mask = src_logits == float('-inf')
#         # 
#         tgt_pos_seg = []
#         for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
#             gt_names = target['gt_names']  # a list of gt box's name
#             all_ids = target['input_ids']
#             pos_seg = []
#             for name in gt_names:
#                 ids = self.tokenizer(name +'.').input_ids[1:-1]
#                 start_i, end_i = search_query_pos(all_ids.tolist(), ids)
#                 assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
#                                 self.tokenizer.decode(all_ids))
#                 pos_seg.append((start_i, end_i))

#             for j in J.tolist():
#                 tgt_pos_seg.append(pos_seg[j])


#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


#         # set positive labels
#         for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
#             target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

#         target_classes_onehot = target_classes_onehot[:,:,:-1]

#         alpha = self.focal_alpha
#         gamma = 2.0

#         loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
#                                      alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
#         loss_ce.masked_fill_(src_mask, 0.)

#         nb_pos = target_classes_onehot.sum(-1)
#         nb_pos[nb_pos == 0] = 1.0

#         loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
#         losses = {'loss_ce': loss_ce}


#         return losses

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes

#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))

#         losses['loss_giou'] = loss_giou.sum() / num_boxes

#         ## calculate the x,y and h,w loss
#         #with torch.no_grad():
#         #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
#         #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

#         return losses

#     def loss_masks(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs

#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)
#         src_masks = outputs["pred_masks"]
#         src_masks = src_masks[src_idx]
#         masks = [t["masks"] for t in targets]
#         # TODO use valid to mask invalid areas due to padding in loss
#         target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
#         target_masks = target_masks.to(src_masks)
#         target_masks = target_masks[tgt_idx]

#         # upsample predictions to the target size
#         src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
#                                 mode="bilinear", align_corners=False)
#         src_masks = src_masks[:, 0].flatten(1)

#         target_masks = target_masks.flatten(1)
#         target_masks = target_masks.view(src_masks.shape)
#         losses = {
#             "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
#             "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
#         }
#         return losses

#     def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
#         assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

#         idx = self._get_src_permutation_idx(indices)
#         pred_obj = outputs["pred_obj"][idx]
#         return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

#     def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
#         """
#         rln_pred_boxes: (bs, n, 4)
#         rln_gt_boxes:   (bs, n, 4)
#         rln_mask: (bs, n)  True = padding / invalid
#         """
#         # flatten
#         src_boxes = rln_pred_boxes.reshape(-1, 4)
#         target_boxes = rln_gt_boxes.reshape(-1, 4)
#         valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

#         # keep only the valid boxes
#         src_boxes = src_boxes[~valid_mask_flat]
#         target_boxes = target_boxes[~valid_mask_flat]

#         # guard against an all-padding batch (edge case)
#         if src_boxes.numel() == 0:
#             # return {
#             #     'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
#             #     'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
#             # }
#             dummy_loss = rln_pred_boxes.sum() * 0.0
#             return {
#                 'loss_rln_bbox': dummy_loss,
#                 'loss_rln_giou': dummy_loss,
#             }

#         # ---------------- L1 ----------------
#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
#         loss_bbox = loss_bbox.sum() / num_rln_boxes

#         # ---------------- GIoU ----------------
#         src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
#         tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

#         giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

#         loss_giou = 1 - torch.diag(giou)
#         loss_giou = loss_giou.sum() / num_rln_boxes

#         return {
#             'loss_rln_bbox': loss_bbox,
#             'loss_rln_giou': loss_giou,
#         }

#     def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
#         """
#           compute loss for relations 
#         """
#         # Flatten outputs for loss calculation.
#         rel_logits = [e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)]
#         # pad_size = 512 - rel_logits.size(-1)
#         # if pad_size > 0:
#         #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

#         # for i, i_rel_logits in enumerate(rel_logits):
#         #     pad_size = 512 - i_rel_logits.size(-1)
#         #     if pad_size > 0:
#         #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
#         #     rel_logits[i] = i_rel_logits

#         rel_tgt_onehot = []
#         for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
#             i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

#             if self.sgg_mode == 'full':
#                 i_rel_tgt_onehot[i_edge_lbl] = 1.0
#             else:
#                 for ii, label in enumerate(i_edge_lbl.tolist()):
#                     name = self.ind_to_predicates[label]
#                     if name == '[UNK]':
#                         continue 

#                     ids = self.tokenizer(name +'.').input_ids[1:-1]
#                     start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
#                     assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

#                     i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
#             rel_tgt_onehot.append(i_rel_tgt_onehot)

#         rel_logits = torch.cat(rel_logits)
#         rel_tgt_onehot = torch.cat(rel_tgt_onehot)
#         rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
#         #rel_num = rel_tgt_onehot.sum()
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(rel_num)
#         rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

#         alpha, gamma = 0.25, 2.0
#         eps = 1e-5 

#         rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
#         rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

#         rel_mask = (rel_logits != float('-inf')).float()
        
#         rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
#         rel_weight[rel_weight == 0] = 1.0

#         pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
#         neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

#         pos_loss = pos_loss.sum() 
#         neg_loss = neg_loss.sum() 

#         loss = (pos_loss + neg_loss) / rel_num

#         losses = dict(loss_rln_labels=loss)
#         with torch.no_grad():
#             losses['loss_rln_labels_pos'] = pos_loss.detach() /  rel_num
#             losses['loss_rln_labels_neg'] = neg_loss.detach() /  rel_num

#         losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

#         return losses

#     def loss_pair_pruning(self, edge_gt_lbls, rln_exist_score, batch_pair_num):
#         """
#         edge_gt_lbls: (b, n)
#         rln_exist_score: (b, n)   # logits
#         batch_pair_num: list[int]
#         """
#         device = rln_exist_score.device
#         b, n = rln_exist_score.shape

#         # -------------------------------------------------
#         # 1. turn GT into binary labels (b, n)
#         # -------------------------------------------------
#         targets = (edge_gt_lbls > 0).float()  # (b, n)

#         # -------------------------------------------------
#         # 2. build the valid mask
#         # -------------------------------------------------
#         valid_mask = torch.zeros((b, n), dtype=torch.bool, device=device)

#         for i in range(b):
#             valid_mask[i, :batch_pair_num[i]] = True

#         # -------------------------------------------------
#         # 3. Focal Loss
#         # -------------------------------------------------
#         logits = rln_exist_score  # (b, n)

#         bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (b, n)

#         pt = torch.exp(-bce_loss)

#         # hyper-parameters, tune later
#         alpha = 0.25  # alpha = 0.5  or 0.75
#         gamma = 2.0

#         focal_loss = alpha * (1 - pt) ** gamma * bce_loss

#         # -------------------------------------------------
#         # 4. mask out padding
#         # -------------------------------------------------
#         focal_loss = focal_loss * valid_mask.float()

#         # -------------------------------------------------
#         # 5. normalize, important
#         # -------------------------------------------------
#         num_valid = valid_mask.sum().clamp(min=1)

#         loss = focal_loss.sum() / num_valid

#         loss_dict = {"loss_pair_pruning": loss}

#         return loss_dict

#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'labels': self.loss_labels,
#             'boxes': self.loss_boxes,
#             'masks': self.loss_masks,
#             'obj_likelihood': self.loss_obj_likelihood,
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     @torch.no_grad()
#     def get_objectness(self, pred_obj):
#         return torch.exp(-self.obj_temp*pred_obj)

#     def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
#         prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
#         mask = (logits != float('-inf')).float()

#         pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
#         neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

#         num_pos = tgt_onehot.sum()
#         if num_pos == 0:
#             return neg_loss.sum()

#         loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

#         return loss

#     def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
#                 global_iter=-1):

#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
            
#              return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

#         """
#         self.global_iter = global_iter

#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
#         device=next(iter(outputs.values())).device

#         # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
#         if 'input_ids' not in targets[0].keys():
#             input_ids = outputs['text_dict']['input_ids']
#             for bid, target in enumerate(targets):
#                 target['input_ids'] = input_ids[bid]
#         indices = self.matcher(outputs_without_aux, targets)
        
#         if outputs_t is not None:
#             outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
#             indices_t = self.matcher(outputs_without_aux_t, targets)
#         else:
#             indices_t = None

#         if return_indices:
#             indices0_copy = indices
#             indices_list = []

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["boxes"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}

#         # =============== dn loss ===============
#         dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

#         if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
#             output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

#             dn_pos_idx = []
#             dn_neg_idx = []
#             for i in range(len(targets)):
#                 if len(targets[i]['boxes']) > 0:
#                     t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
#                     t = t.unsqueeze(0).repeat(scalar, 1)
#                     tgt_idx = t.flatten()
#                     output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
#                     output_idx = output_idx.flatten()
#                 else:
#                     output_idx = tgt_idx = torch.tensor([]).long().cuda()

#                 dn_pos_idx.append((output_idx, tgt_idx))
#                 dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

#             output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
#             l_dict = {}
#             for loss in self.losses:
#                 if 'obj_likelihood' == loss:
#                     continue 
#                 kwargs = {}
#                 if 'labels' in loss:
#                     kwargs = {'log': False}
#                 l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

#             l_dict = {k + f'_dn': v for k, v in l_dict.items()}
#             losses.update(l_dict)
#         else:
#             pass

#         # =============== standard loss ===============
#         for loss in self.losses:
#             kwargs = {}
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

#         # =============== aux loss ===============
#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for idx, aux_output in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_output, targets)
#                 if outputs_t is not None:
#                     aux_outputs_t = outputs_t['aux_outputs'][idx]
#                     indices_t = self.matcher(aux_outputs_t, targets)

#                 if return_indices:
#                     indices_list.append(indices)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}

#                     l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#                 if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
#                     aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
#                     l_dict={}
#                     for loss in self.losses:
#                         kwargs = {}
#                         if 'labels' in loss:
#                             kwargs = {'log': False}

#                         l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

#                     l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#                 else:
#                     pass

#         # =============== interm loss ===============
#         # interm_outputs loss
#         if 'interm_outputs' in outputs:
#             interm_outputs = outputs['interm_outputs']
#             indices = self.matcher(interm_outputs, targets)

#             if outputs_t is not None:
#                 interm_outputs_t = outputs_t['interm_outputs']
#                 indices_t = self.matcher(interm_outputs_t, targets)

#             if return_indices:
#                 indices_list.append(indices)
#             for loss in self.losses:
#                 if loss == 'masks':
#                     # Intermediate masks losses are too costly to compute, we ignore them.
#                     continue
#                 if 'obj_likelihood' == loss:
#                     continue 
#                 kwargs = {}
#                 if loss == 'labels':
#                     # Logging is enabled only for the last layer
#                     kwargs = {'log': False}

#                 l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
#                 l_dict = {k + f'_interm': v for k, v in l_dict.items()}
#                 losses.update(l_dict)

#         # =============== enc_outputs loss ===============
#         if 'enc_outputs' in outputs:
#             for i, enc_outputs in enumerate(outputs['enc_outputs']):
#                 indices = self.matcher(enc_outputs, targets)
#                 if return_indices:
#                     indices_list.append(indices)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     if 'obj_likelihood' == loss:
#                         continue 
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}

#                     l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         # =============== rln_decoder loss ===============
#         rln_out = outputs['rln_out']
#         rln_text_dict = outputs['rel_text_dict']
#         rln_query_mask = rln_out['rln_query_mask']
#         batch_pair_num = rln_out['batch_pair_num']
#         edge_gt_lbls_padded = outputs['edge_gt_lbls']
#         edge_gt_boxes = outputs['edge_gt_boxes']

#         rln_mask = (edge_gt_lbls_padded == 0)
#         edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls_padded, batch_pair_num)]

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_rln_boxes = sum(len(t["edges"]) for t in targets)
#         num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_rln_boxes)
#         num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

#         # pair pruning loss
#         pair_pruning_loss = self.loss_pair_pruning(edge_gt_lbls_padded, rln_out['rln_exist_score'], batch_pair_num)
#         losses.update(pair_pruning_loss)

#         # interm class/box
#         interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
#         interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
#         interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
#         interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
#         losses.update(interm_loss_rln_boxes_dict)
#         losses.update(interm_loss_rln_labels_dict)

#         # aux class/box
#         for idx, aux_output in enumerate(rln_out['aux_outputs']):
#             aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
#             aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
#             aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
#             aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
#             losses.update(aux_loss_rln_boxes_dict)
#             losses.update(aux_loss_rln_labels_dict)

#         # standard class / box
#         loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
#         loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
#         losses.update(loss_rln_boxes_dict)
#         losses.update(loss_rln_labels_dict)

#         if return_indices:
#             indices_list.append(indices0_copy)
#             return losses, indices_list

#         return losses

#     def prep_for_dn(self,dn_meta):
#         output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
#         num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
#         assert pad_size % num_dn_groups==0
#         single_pad=pad_size//num_dn_groups

#         return output_known_lbs_bboxes,single_pad,num_dn_groups



# class OvSGTRRelDecoderPairPruningCriterionV2(nn.Module):
#     """ This class computes the loss for Conditional DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """
#     def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
#                 **kwargs):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of object categories, omitting the special no-object category
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#             focal_alpha: alpha in Focal Loss
#         """
#         super().__init__()
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
#         self.focal_alpha = focal_alpha
#         self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

#         self.tokenizer = kwargs.get("tokenizer", None)
#         self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
#         self.global_iter = -1
#         hidden_dim = kwargs.get("hidden_dim", 256)
#         self.min_obj=-hidden_dim*math.log(0.9)
#         self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
#         self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
#         self.obj_threshold = kwargs.get("obj_threshold", 0.5)

#         self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
#         self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

#         self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (Binary focal loss)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         idx = self._get_src_permutation_idx(indices)

#         src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
#         src_mask = src_logits == float('-inf')
#         # 
#         tgt_pos_seg = []
#         for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
#             gt_names = target['gt_names']  # a list of gt box's name
#             all_ids = target['input_ids']
#             pos_seg = []
#             for name in gt_names:
#                 ids = self.tokenizer(name +'.').input_ids[1:-1]
#                 start_i, end_i = search_query_pos(all_ids.tolist(), ids)
#                 assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
#                                 self.tokenizer.decode(all_ids))
#                 pos_seg.append((start_i, end_i))

#             for j in J.tolist():
#                 tgt_pos_seg.append(pos_seg[j])


#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


#         # set positive labels
#         for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
#             target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

#         target_classes_onehot = target_classes_onehot[:,:,:-1]

#         alpha = self.focal_alpha
#         gamma = 2.0

#         loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
#                                      alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
#         loss_ce.masked_fill_(src_mask, 0.)

#         nb_pos = target_classes_onehot.sum(-1)
#         nb_pos[nb_pos == 0] = 1.0

#         loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
#         losses = {'loss_ce': loss_ce}


#         return losses

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes

#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))

#         losses['loss_giou'] = loss_giou.sum() / num_boxes

#         ## calculate the x,y and h,w loss
#         #with torch.no_grad():
#         #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
#         #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

#         return losses

#     def loss_masks(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs

#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)
#         src_masks = outputs["pred_masks"]
#         src_masks = src_masks[src_idx]
#         masks = [t["masks"] for t in targets]
#         # TODO use valid to mask invalid areas due to padding in loss
#         target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
#         target_masks = target_masks.to(src_masks)
#         target_masks = target_masks[tgt_idx]

#         # upsample predictions to the target size
#         src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
#                                 mode="bilinear", align_corners=False)
#         src_masks = src_masks[:, 0].flatten(1)

#         target_masks = target_masks.flatten(1)
#         target_masks = target_masks.view(src_masks.shape)
#         losses = {
#             "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
#             "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
#         }
#         return losses

#     def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
#         assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

#         idx = self._get_src_permutation_idx(indices)
#         pred_obj = outputs["pred_obj"][idx]
#         return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

#     def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
#         """
#         rln_pred_boxes: (bs, n, 4)
#         rln_gt_boxes:   (bs, n, 4)
#         rln_mask: (bs, n)  True = padding / invalid
#         """
#         # flatten
#         src_boxes = rln_pred_boxes.reshape(-1, 4)
#         target_boxes = rln_gt_boxes.reshape(-1, 4)
#         valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

#         # keep only the valid boxes
#         src_boxes = src_boxes[~valid_mask_flat]
#         target_boxes = target_boxes[~valid_mask_flat]

#         # guard against an all-padding batch (edge case)
#         if src_boxes.numel() == 0:
#             # return {
#             #     'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
#             #     'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
#             # }
#             dummy_loss = rln_pred_boxes.sum() * 0.0
#             return {
#                 'loss_rln_bbox': dummy_loss,
#                 'loss_rln_giou': dummy_loss,
#             }

#         # ---------------- L1 ----------------
#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
#         loss_bbox = loss_bbox.sum() / num_rln_boxes

#         # ---------------- GIoU ----------------
#         src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
#         tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

#         giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

#         loss_giou = 1 - torch.diag(giou)
#         loss_giou = loss_giou.sum() / num_rln_boxes

#         return {
#             'loss_rln_bbox': loss_bbox,
#             'loss_rln_giou': loss_giou,
#         }

#     def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
#         """
#           compute loss for relations 
#         """
#         # Flatten outputs for loss calculation.
#         rel_logits = [e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)]
#         # pad_size = 512 - rel_logits.size(-1)
#         # if pad_size > 0:
#         #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

#         # for i, i_rel_logits in enumerate(rel_logits):
#         #     pad_size = 512 - i_rel_logits.size(-1)
#         #     if pad_size > 0:
#         #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
#         #     rel_logits[i] = i_rel_logits

#         rel_tgt_onehot = []
#         for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
#             i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

#             if self.sgg_mode == 'full':
#                 i_rel_tgt_onehot[i_edge_lbl] = 1.0
#             else:
#                 for ii, label in enumerate(i_edge_lbl.tolist()):
#                     name = self.ind_to_predicates[label]
#                     if name == '[UNK]':
#                         continue 

#                     ids = self.tokenizer(name +'.').input_ids[1:-1]
#                     start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
#                     assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

#                     i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
#             rel_tgt_onehot.append(i_rel_tgt_onehot)

#         rel_logits = torch.cat(rel_logits)
#         rel_tgt_onehot = torch.cat(rel_tgt_onehot)
#         rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
#         #rel_num = rel_tgt_onehot.sum()
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(rel_num)
#         rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

#         alpha, gamma = 0.25, 2.0
#         eps = 1e-5 

#         rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
#         rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

#         rel_mask = (rel_logits != float('-inf')).float()
        
#         rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
#         rel_weight[rel_weight == 0] = 1.0

#         pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
#         neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

#         pos_loss = pos_loss.sum() 
#         neg_loss = neg_loss.sum() 

#         loss = (pos_loss + neg_loss) / rel_num

#         losses = dict(loss_rln_labels=loss)
#         with torch.no_grad():
#             losses['loss_rln_labels_pos'] = pos_loss.detach() /  rel_num
#             losses['loss_rln_labels_neg'] = neg_loss.detach() /  rel_num

#         losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

#         return losses

#     def loss_pair_pruning(self, edge_gt_lbls, rln_exist_score, batch_pair_num):
#         """
#         edge_gt_lbls: (b, n)
#         rln_exist_score: (b, n)   # logits
#         batch_pair_num: list[int]
#         """
#         device = rln_exist_score.device
#         b, n = rln_exist_score.shape

#         # -------------------------------------------------
#         # 1. turn GT into binary labels (b, n)
#         # -------------------------------------------------
#         targets = (edge_gt_lbls > 0).float()  # (b, n)

#         # -------------------------------------------------
#         # 2. build the valid mask
#         # -------------------------------------------------
#         valid_mask = torch.zeros((b, n), dtype=torch.bool, device=device)

#         for i in range(b):
#             valid_mask[i, :batch_pair_num[i]] = True

#         # -------------------------------------------------
#         # 3. Focal Loss
#         # -------------------------------------------------
#         logits = rln_exist_score  # (b, n)

#         bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (b, n)

#         pt = torch.exp(-bce_loss)

#         # hyper-parameters, tune later
#         alpha = 0.25  # alpha = 0.5  or 0.75
#         gamma = 2.0

#         focal_loss = alpha * (1 - pt) ** gamma * bce_loss

#         # -------------------------------------------------
#         # 4. mask out padding
#         # -------------------------------------------------
#         focal_loss = focal_loss * valid_mask.float()

#         # -------------------------------------------------
#         # 5. normalize, important
#         # -------------------------------------------------
#         num_valid = valid_mask.sum().clamp(min=1)

#         loss = focal_loss.sum() / num_valid

#         loss_dict = {"loss_pair_pruning": loss}

#         return loss_dict

#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         loss_map = {
#             'labels': self.loss_labels,
#             'boxes': self.loss_boxes,
#             'masks': self.loss_masks,
#             'obj_likelihood': self.loss_obj_likelihood,
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     @torch.no_grad()
#     def get_objectness(self, pred_obj):
#         return torch.exp(-self.obj_temp*pred_obj)

#     def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
#         prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
#         mask = (logits != float('-inf')).float()

#         pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
#         neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

#         num_pos = tgt_onehot.sum()
#         if num_pos == 0:
#             return neg_loss.sum()

#         loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

#         return loss

#     def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
#                 global_iter=-1):

#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
            
#              return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

#         """
#         self.global_iter = global_iter

#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
#         device=next(iter(outputs.values())).device

#         # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
#         if 'input_ids' not in targets[0].keys():
#             input_ids = outputs['text_dict']['input_ids']
#             for bid, target in enumerate(targets):
#                 target['input_ids'] = input_ids[bid]
#         indices = self.matcher(outputs_without_aux, targets)
        
#         if outputs_t is not None:
#             outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
#             indices_t = self.matcher(outputs_without_aux_t, targets)
#         else:
#             indices_t = None

#         if return_indices:
#             indices0_copy = indices
#             indices_list = []

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["boxes"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         # Compute all the requested losses
#         losses = {}

#         # =============== dn loss ===============
#         dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

#         if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
#             output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

#             dn_pos_idx = []
#             dn_neg_idx = []
#             for i in range(len(targets)):
#                 if len(targets[i]['boxes']) > 0:
#                     t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
#                     t = t.unsqueeze(0).repeat(scalar, 1)
#                     tgt_idx = t.flatten()
#                     output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
#                     output_idx = output_idx.flatten()
#                 else:
#                     output_idx = tgt_idx = torch.tensor([]).long().cuda()

#                 dn_pos_idx.append((output_idx, tgt_idx))
#                 dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

#             output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
#             l_dict = {}
#             for loss in self.losses:
#                 if 'obj_likelihood' == loss:
#                     continue 
#                 kwargs = {}
#                 if 'labels' in loss:
#                     kwargs = {'log': False}
#                 l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

#             l_dict = {k + f'_dn': v for k, v in l_dict.items()}
#             losses.update(l_dict)
#         else:
#             pass

#         # =============== standard loss ===============
#         for loss in self.losses:
#             kwargs = {}
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

#         # =============== aux loss ===============
#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for idx, aux_output in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_output, targets)
#                 if outputs_t is not None:
#                     aux_outputs_t = outputs_t['aux_outputs'][idx]
#                     indices_t = self.matcher(aux_outputs_t, targets)

#                 if return_indices:
#                     indices_list.append(indices)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}

#                     l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#                 if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
#                     aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
#                     l_dict={}
#                     for loss in self.losses:
#                         kwargs = {}
#                         if 'labels' in loss:
#                             kwargs = {'log': False}

#                         l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

#                     l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#                 else:
#                     pass

#         # =============== interm loss ===============
#         # interm_outputs loss
#         if 'interm_outputs' in outputs:
#             interm_outputs = outputs['interm_outputs']
#             indices = self.matcher(interm_outputs, targets)

#             if outputs_t is not None:
#                 interm_outputs_t = outputs_t['interm_outputs']
#                 indices_t = self.matcher(interm_outputs_t, targets)

#             if return_indices:
#                 indices_list.append(indices)
#             for loss in self.losses:
#                 if loss == 'masks':
#                     # Intermediate masks losses are too costly to compute, we ignore them.
#                     continue
#                 if 'obj_likelihood' == loss:
#                     continue 
#                 kwargs = {}
#                 if loss == 'labels':
#                     # Logging is enabled only for the last layer
#                     kwargs = {'log': False}

#                 l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
#                 l_dict = {k + f'_interm': v for k, v in l_dict.items()}
#                 losses.update(l_dict)

#         # =============== enc_outputs loss ===============
#         if 'enc_outputs' in outputs:
#             for i, enc_outputs in enumerate(outputs['enc_outputs']):
#                 indices = self.matcher(enc_outputs, targets)
#                 if return_indices:
#                     indices_list.append(indices)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     if 'obj_likelihood' == loss:
#                         continue 
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs = {'log': False}

#                     l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)

#         # =============== rln_decoder loss ===============
#         rln_out = outputs['rln_out']
#         rln_text_dict = outputs['rel_text_dict']
#         rln_query_mask = rln_out['rln_query_mask']
#         batch_pair_num = rln_out['batch_pair_num']
#         edge_gt_lbls_padded = outputs['edge_gt_lbls']
#         edge_gt_boxes = outputs['edge_gt_boxes']

#         rln_mask = (edge_gt_lbls_padded == 0)
#         edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls_padded, batch_pair_num)]

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_rln_boxes = sum(len(t["edges"]) for t in targets)
#         num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_rln_boxes)
#         num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

#         # pair pruning loss
#         pair_pruning_loss = self.loss_pair_pruning(rln_out['pair_selector_rln_gt_labels'], rln_out['rln_exist_score'], rln_out['pair_selector_batch_pair_num'])
#         losses.update(pair_pruning_loss)

#         # interm class/box
#         interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
#         interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
#         interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
#         interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
#         losses.update(interm_loss_rln_boxes_dict)
#         losses.update(interm_loss_rln_labels_dict)

#         # aux class/box
#         for idx, aux_output in enumerate(rln_out['aux_outputs']):
#             aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
#             aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
#             aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
#             aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
#             losses.update(aux_loss_rln_boxes_dict)
#             losses.update(aux_loss_rln_labels_dict)

#         # standard class / box
#         loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
#         loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
#         losses.update(loss_rln_boxes_dict)
#         losses.update(loss_rln_labels_dict)

#         if return_indices:
#             indices_list.append(indices0_copy)
#             return losses, indices_list

#         return losses

#     def prep_for_dn(self,dn_meta):
#         output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
#         num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
#         assert pad_size % num_dn_groups==0
#         single_pad=pad_size//num_dn_groups

#         return output_known_lbs_bboxes,single_pad,num_dn_groups



class OvSGTRRelDecoderPairPruningCriterionV2(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

        self.rln_pos_weight = kwargs.get("rln_pos_weight", 1.0)
        self.rln_neg_weight = kwargs.get("rln_neg_weight", 1.0)
        self.rln_loss_norm_mode = kwargs.get("rln_loss_norm_mode", "joint")   # 'joint' | 'separate'
        self.rln_loss_alpha = kwargs.get("rln_loss_alpha", 0.5)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
        """
        rln_pred_boxes: (bs, n, 4)
        rln_gt_boxes:   (bs, n, 4)
        rln_mask: (bs, n)  True = padding / invalid
        """
        # flatten
        src_boxes = rln_pred_boxes.reshape(-1, 4)
        target_boxes = rln_gt_boxes.reshape(-1, 4)
        valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

        # keep only the valid boxes
        src_boxes = src_boxes[~valid_mask_flat]
        target_boxes = target_boxes[~valid_mask_flat]

        # guard against an all-padding batch (edge case)
        if src_boxes.numel() == 0:
            # return {
            #     'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
            #     'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
            # }
            dummy_loss = rln_pred_boxes.sum() * 0.0
            return {
                'loss_rln_bbox': dummy_loss,
                'loss_rln_giou': dummy_loss,
            }

        # ---------------- L1 ----------------
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_rln_boxes

        # ---------------- GIoU ----------------
        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

        giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.sum() / num_rln_boxes

        return {
            'loss_rln_bbox': loss_bbox,
            'loss_rln_giou': loss_giou,
        }

    # def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
    def loss_rln_labels(self, rln_pred_logits, edge_lbls, pair_mask, rel_text_dict, pos_weight=1.0, neg_weight=1.0,
                        norm_mode='joint', loss_alpha=0.5):
        """
          compute loss for relations 
        """
        # Flatten outputs for loss calculation.
        rel_logits = [e[~x] for e, x in zip(rln_pred_logits, pair_mask)]
        # pad_size = 512 - rel_logits.size(-1)
        # if pad_size > 0:
        #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

        # for i, i_rel_logits in enumerate(rel_logits):
        #     pad_size = 512 - i_rel_logits.size(-1)
        #     if pad_size > 0:
        #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
        #     rel_logits[i] = i_rel_logits

        rel_tgt_onehot = []
        for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
            i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

            if self.sgg_mode == 'full':
                i_rel_tgt_onehot[i_edge_lbl] = 1.0
            else:
                for ii, label in enumerate(i_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        rel_logits = torch.cat(rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)
        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 

        rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
        rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

        rel_mask = (rel_logits != float('-inf')).float()
        
        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        # original OvSGTR per-token focal (both divided by rel_weight = #predicate tokens/pair)
        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_sum = pos_loss.sum() * pos_weight
        neg_sum = neg_loss.sum() * neg_weight

        if norm_mode == 'separate':
            # Normalize each side by its own PAIR count, then combine with loss_alpha.
            # (Positive pairs also contribute a little to neg_sum -> treated as negligible.)
            # NOTE: neg_loss keeps the original /rel_weight, so each NEGATIVE pair still
            # weighs ~T units (T = #relation-text tokens). Hence loss_alpha is NOT
            # 0.5-balanced, and since T differs between pretrain/finetune (relation vocab),
            # loss_alpha generally needs re-tuning per stage.
            pos_num = ((rel_tgt_onehot * rel_mask).sum(dim=1) > 0).float().sum().clamp(min=1.0)            # #positive pairs
            neg_num = torch.clamp(torch.as_tensor(float(rel_num), device=rel_prob.device) - pos_num, min=1.0)  # #negative pairs
            loss = loss_alpha * (pos_sum / pos_num) + (1.0 - loss_alpha) * (neg_sum / neg_num)
            # log the WEIGHTED contributions (each = a term of `loss`), so they are
            # directly comparable to the 'joint'/V6 logs (pos_frac*mean_pos etc.)
            pos_log = (loss_alpha * pos_sum / pos_num).detach()
            neg_log = ((1.0 - loss_alpha) * neg_sum / neg_num).detach()
        else:  # 'joint': exact original OvSGTR / V6 behaviour
            loss = (pos_sum + neg_sum) / rel_num
            pos_log = pos_sum.detach() / rel_num
            neg_log = neg_sum.detach() / rel_num

        losses = dict(loss_rln_labels=loss)
        with torch.no_grad():
            losses['loss_rln_labels_pos'] = pos_log
            losses['loss_rln_labels_neg'] = neg_log

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

        return losses

    # def loss_pair_pruning(self, edge_gt_lbls, rln_exist_score, batch_pair_num):
    def loss_pair_pruning(self, edge_gt_lbls, rln_exist_score, pair_mask):
        """
        edge_gt_lbls: (b, n)
        rln_exist_score: (b, n)   # logits
        batch_pair_num: list[int]
        """
        device = rln_exist_score.device
        b, n = rln_exist_score.shape

        # -------------------------------------------------
        # 1. turn GT into binary labels (b, n)
        # -------------------------------------------------
        targets = (edge_gt_lbls > 0).float()  # (b, n)

        # -------------------------------------------------
        # 2. build the valid mask
        # -------------------------------------------------
        valid_mask = torch.zeros((b, n), dtype=torch.bool, device=device)

        for i in range(b):
            # valid_mask[i, :batch_pair_num[i]] = True
            valid_mask[i, ~pair_mask[i]] = True

        # -------------------------------------------------
        # 3. Focal Loss
        # -------------------------------------------------
        # The selector can be untrained (loss weight 0) -> its random-init
        # logits may hit inf under amp; clamp saturates the BCE identically
        # for anything beyond +-50 so trained selectors are unaffected.
        logits = rln_exist_score.float().clamp(min=-50.0, max=50.0)  # (b, n)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (b, n)

        pt = torch.exp(-bce_loss)

        # hyper-parameters, tune later
        alpha = 0.25  # alpha = 0.5  or 0.75
        gamma = 2.0

        focal_loss = alpha * (1 - pt) ** gamma * bce_loss

        # -------------------------------------------------
        # 4. mask out padding (torch.where, NOT multiply: nan/inf at padded
        #    positions would survive `nan * 0` and poison the sum)
        # -------------------------------------------------
        focal_loss = torch.where(valid_mask, focal_loss, torch.zeros_like(focal_loss))

        # -------------------------------------------------
        # 5. normalize, important
        # -------------------------------------------------
        num_valid = valid_mask.sum().clamp(min=1)

        loss = focal_loss.sum() / num_valid

        loss_dict = {"loss_pair_pruning": loss}

        return loss_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        if 'input_ids' not in targets[0].keys():
            input_ids = outputs['text_dict']['input_ids']
            for bid, target in enumerate(targets):
                target['input_ids'] = input_ids[bid]
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # =============== dn loss ===============
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        # =============== standard loss ===============
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # =============== aux loss ===============
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_output in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_output, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # =============== interm loss ===============
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # =============== enc_outputs loss ===============
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # =============== rln_decoder loss ===============
        rln_out = outputs['rln_out']
        rln_text_dict = outputs['rel_text_dict']
        rln_query_mask = rln_out['rln_query_mask']
        batch_pair_num = rln_out['batch_pair_num']
        edge_gt_lbls_padded = outputs['edge_gt_lbls']
        edge_gt_boxes = outputs['edge_gt_boxes']

        rln_mask = ((edge_gt_lbls_padded == 0) | (edge_gt_lbls_padded == -1))
        # Pairs supervised by the relation-label loss. Defaults to pair_mask
        # (every valid pair) for backward compatibility; with the model's
        # rln_loss_sample_mode='balanced' this is a V6-style balanced subset.
        # The box loss above (rln_mask) and the decoder forward are unaffected.
        rln_label_loss_mask = rln_out.get('rln_loss_mask', rln_out['pair_mask'])
        # edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls_padded, batch_pair_num)]
        edge_gt_lbls = [e[~x] for e, x in zip(edge_gt_lbls_padded, rln_label_loss_mask)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_rln_boxes = sum(len(t["edges"]) for t in targets)
        num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_rln_boxes)
        num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

        # pair pruning loss
        # pair_pruning_loss = self.loss_pair_pruning(rln_out['pair_selector_rln_gt_labels'], rln_out['rln_exist_score'], rln_out['pair_selector_batch_pair_num'])
        pair_pruning_loss = self.loss_pair_pruning(rln_out['pair_selector_rln_gt_labels'], rln_out['rln_exist_score'], rln_out['pair_selector_pair_mask'])
        losses.update(pair_pruning_loss)

        # interm class/box
        interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        # interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, rln_label_loss_mask, rln_text_dict, self.rln_pos_weight, self.rln_neg_weight, self.rln_loss_norm_mode, self.rln_loss_alpha)
        interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
        interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
        losses.update(interm_loss_rln_boxes_dict)
        losses.update(interm_loss_rln_labels_dict)

        # aux class/box
        for idx, aux_output in enumerate(rln_out['aux_outputs']):
            aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
            # aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
            aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, rln_label_loss_mask, rln_text_dict, self.rln_pos_weight, self.rln_neg_weight, self.rln_loss_norm_mode, self.rln_loss_alpha)
            aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
            aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
            losses.update(aux_loss_rln_boxes_dict)
            losses.update(aux_loss_rln_labels_dict)

        # standard class / box
        loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        # loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, rln_label_loss_mask, rln_text_dict, self.rln_pos_weight, self.rln_neg_weight, self.rln_loss_norm_mode, self.rln_loss_alpha)
        losses.update(loss_rln_boxes_dict)
        losses.update(loss_rln_labels_dict)

        # =============== distillation (frozen teacher present) ===============
        # V6-style anti-forgetting distillation at finetune. Requires the
        # teacher to have been fed the student's pool_spec so that pools/pairs
        # align slot-by-slot (engine does this).
        if outputs_t is not None and 'loss_distill' in self.weight_dict:
            rln_out_t = outputs_t['rln_out']
            assert torch.equal(edge_gt_lbls_padded, outputs_t['edge_gt_lbls']), \
                "student/teacher pair labels diverged -- pool_spec sharing is broken"
            for idx, (aux_output, aux_output_t) in enumerate(zip(rln_out['aux_outputs'], rln_out_t['aux_outputs'])):
                aux_distill_dict = self.loss_rln_distill(aux_output['rln_hs'], aux_output_t['rln_hs'], edge_gt_lbls_padded, rln_label_loss_mask)
                losses.update({k + f'_{idx}': v for k, v in aux_distill_dict.items()})
            losses.update(self.loss_rln_distill(rln_out['rln_hs'], rln_out_t['rln_hs'], edge_gt_lbls_padded, rln_label_loss_mask))
        if outputs_t is not None and 'loss_encoder_distill' in self.weight_dict:
            enc, enc_t = outputs['encoder_output'], outputs_t['encoder_output']
            losses.update(self.loss_encoder_distill(
                enc['memory'], enc['memory_text'],
                enc_t['memory'], enc_t['memory_text'],
                enc['memory_mask'], enc['memory_text_mask']))

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def loss_rln_distill(self, rln_hs, rln_hs_t, edge_gt_lbls_padded, rln_label_loss_mask):
        """V6-faithful unsupervised relation distillation (see CriterionV6
        loss_rln_distill): L1 between the student's and the frozen teacher's
        relation-decoder hidden states, restricted to the NEGATIVE (label==0)
        pairs of the loss-sampled rel batch. Positives learn from VG labels;
        negatives (where novel relations hide and the label loss pushes toward
        background) stay anchored to the pretrain teacher. Normalized by the
        total sampled pair count (pos+neg) all-reduced across workers, as V6.

        rln_hs / rln_hs_t: (b, K, d). edge_gt_lbls_padded: (b, K) long,
        -1 pad / 0 negative / >0 positive. rln_label_loss_mask: (b, K) bool,
        True = excluded from the label loss.
        """
        keep = ~rln_label_loss_mask
        rel_num = keep.sum().float()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        ids = keep & (edge_gt_lbls_padded == 0)
        loss_distill = F.l1_loss(rln_hs[ids], rln_hs_t[ids].detach(), reduction='sum') / rel_num
        return dict(loss_distill=loss_distill)

    def loss_encoder_distill(self, memory, memory_text, memory_t, memory_text_t, memory_mask, memory_text_mask):
        # Ported from CriterionV6.loss_encoder_distill: masked-mean L1 between
        # student and frozen-teacher fused-encoder outputs (image + text).
        image_valid = (~memory_mask).float()
        text_valid = (~memory_text_mask).float()

        image_loss = F.l1_loss(memory, memory_t.detach(), reduction='none').mean(dim=-1)  # (bs, num_tokens)
        text_loss = F.l1_loss(memory_text, memory_text_t.detach(), reduction='none').mean(dim=-1)  # (bs, text_len)

        encoder_image_distill_loss = (image_loss * image_valid).sum() / image_valid.sum().clamp(min=1)
        encoder_text_distill_loss = (text_loss * text_valid).sum() / text_valid.sum().clamp(min=1)

        return dict(loss_encoder_distill=(encoder_image_distill_loss + encoder_text_distill_loss))

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups




class OvSGTRRelDecoderPairPruningCriterionV4(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

        self.pair_loss_use_prob = kwargs.get("pair_loss_with_box", False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
        """
        rln_pred_boxes: (bs, n, 4)
        rln_gt_boxes:   (bs, n, 4)
        rln_mask: (bs, n)  True = padding / invalid
        """
        # flatten
        src_boxes = rln_pred_boxes.reshape(-1, 4)
        target_boxes = rln_gt_boxes.reshape(-1, 4)
        valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

        # keep only the valid boxes
        src_boxes = src_boxes[~valid_mask_flat]
        target_boxes = target_boxes[~valid_mask_flat]

        # guard against an all-padding batch (edge case)
        if src_boxes.numel() == 0:
            # return {
            #     'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
            #     'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
            # }
            dummy_loss = rln_pred_boxes.sum() * 0.0
            return {
                'loss_rln_bbox': dummy_loss,
                'loss_rln_giou': dummy_loss,
            }

        # ---------------- L1 ----------------
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_rln_boxes

        # ---------------- GIoU ----------------
        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

        giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.sum() / num_rln_boxes

        return {
            'loss_rln_bbox': loss_bbox,
            'loss_rln_giou': loss_giou,
        }

    def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
        """
          compute loss for relations 
        """
        # Flatten outputs for loss calculation.
        rel_logits = [e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)]
        # pad_size = 512 - rel_logits.size(-1)
        # if pad_size > 0:
        #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

        # for i, i_rel_logits in enumerate(rel_logits):
        #     pad_size = 512 - i_rel_logits.size(-1)
        #     if pad_size > 0:
        #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
        #     rel_logits[i] = i_rel_logits

        rel_tgt_onehot = []
        for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
            i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

            if self.sgg_mode == 'full':
                i_rel_tgt_onehot[i_edge_lbl] = 1.0
            else:
                for ii, label in enumerate(i_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        rel_logits = torch.cat(rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)
        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 

        rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
        rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

        rel_mask = (rel_logits != float('-inf')).float()
        
        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum() 
        neg_loss = neg_loss.sum() 

        loss = (pos_loss + neg_loss) / rel_num

        losses = dict(loss_rln_labels=loss)
        with torch.no_grad():
            losses['loss_rln_labels_pos'] = pos_loss.detach() /  rel_num
            losses['loss_rln_labels_neg'] = neg_loss.detach() /  rel_num

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

        return losses

    def loss_pair_pruning(self, rln_gt_labels_all, rln_exist_logits, batch_pair_num):
        """
        rln_gt_labels_all: (b, n * n)
        rln_exist_logits: (b, n * n)   # logits
        batch_pair_num: list[int]
        """
        device = rln_exist_logits.device
        b, nn = rln_exist_logits.shape
        n = int(nn ** 0.5)

        # -------------------------------------------------
        # 1. turn GT into binary labels (b, n)
        # -------------------------------------------------
        targets = (rln_gt_labels_all > 0).float()  # (b, n)

        # -------------------------------------------------
        # 2. build the valid mask
        # -------------------------------------------------
        valid_mask = torch.zeros(b, n, n).to(device)
        valid_mask[:, torch.arange(n), torch.arange(n)] = 1
        valid_mask = valid_mask.view(b, n * n)
        # valid_mask = torch.zeros((b, n), dtype=torch.bool, device=device)

        # for i in range(b):
        #     valid_mask[i, :batch_pair_num[i]] = True

        # -------------------------------------------------
        # 3. Focal Loss
        # -------------------------------------------------
        logits = rln_exist_logits  # (b, n)

        if self.pair_loss_use_prob:
            prob = logits
            bce_loss = F.binary_cross_entropy(prob, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (b, n)

        pt = torch.exp(-bce_loss)

        # hyper-parameters, tune later
        alpha = 0.25  # alpha = 0.5  or 0.75
        gamma = 2.0

        focal_loss = alpha * (1 - pt) ** gamma * bce_loss

        # -------------------------------------------------
        # 4. mask out padding
        # -------------------------------------------------
        focal_loss = focal_loss * valid_mask.float()

        # -------------------------------------------------
        # 5. normalize, important
        # -------------------------------------------------
        num_valid = valid_mask.sum().clamp(min=1)

        loss = focal_loss.sum() / num_valid

        loss_dict = {"loss_pair_pruning": loss}

        return loss_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        if 'input_ids' not in targets[0].keys():
            input_ids = outputs['text_dict']['input_ids']
            for bid, target in enumerate(targets):
                target['input_ids'] = input_ids[bid]
        indices = self.matcher(outputs_without_aux, targets)
        
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # =============== dn loss ===============
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        # =============== standard loss ===============
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # =============== aux loss ===============
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_output in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_output, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # =============== interm loss ===============
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # =============== enc_outputs loss ===============
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # =============== rln_decoder loss ===============
        rln_out = outputs['rln_out']
        rln_text_dict = outputs['rel_text_dict']
        rln_query_mask = rln_out['rln_query_mask']
        batch_pair_num = rln_out['batch_pair_num']
        edge_gt_lbls_padded = outputs['edge_gt_lbls']
        edge_gt_boxes = outputs['edge_gt_boxes']

        rln_mask = (edge_gt_lbls_padded == 0)
        edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls_padded, batch_pair_num)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_rln_boxes = sum(len(t["edges"]) for t in targets)
        num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_rln_boxes)
        num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

        # pair pruning loss
        pair_pruning_loss = self.loss_pair_pruning(rln_out['rln_gt_labels_all'], rln_out['rln_exist_logits'], batch_pair_num)
        losses.update(pair_pruning_loss)

        # interm class/box
        interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
        interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
        losses.update(interm_loss_rln_boxes_dict)
        losses.update(interm_loss_rln_labels_dict)

        # aux class/box
        for idx, aux_output in enumerate(rln_out['aux_outputs']):
            aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
            aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
            aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
            aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
            losses.update(aux_loss_rln_boxes_dict)
            losses.update(aux_loss_rln_labels_dict)

        # standard class / box
        loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        losses.update(loss_rln_boxes_dict)
        losses.update(loss_rln_labels_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class OvSGTRRelDecoderPairPruningCriterionV6(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

        self.use_prob_distill = kwargs.get('use_prob_distill', False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
        """
        rln_pred_boxes: (bs, n, 4)
        rln_gt_boxes:   (bs, n, 4)
        rln_mask: (bs, n)  True = padding / invalid
        """
        # flatten
        src_boxes = rln_pred_boxes.reshape(-1, 4)
        target_boxes = rln_gt_boxes.reshape(-1, 4)
        valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

        # keep only the valid boxes
        src_boxes = src_boxes[~valid_mask_flat]
        target_boxes = target_boxes[~valid_mask_flat]

        # guard against an all-padding batch (edge case)
        if src_boxes.numel() == 0:
            # return {
            #     'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
            #     'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
            # }
            dummy_loss = rln_pred_boxes.sum() * 0.0
            return {
                'loss_rln_bbox': dummy_loss,
                'loss_rln_giou': dummy_loss,
            }

        # ---------------- L1 ----------------
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_rln_boxes

        # ---------------- GIoU ----------------
        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

        giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.sum() / num_rln_boxes

        return {
            'loss_rln_bbox': loss_bbox,
            'loss_rln_giou': loss_giou,
        }

    def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
        """
          compute loss for relations 
        """
        def get_ddp_average(num: int, device):
            num = torch.as_tensor(num, device=device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num)
            num = torch.clamp(num / get_world_size(), min=1).item()
            return num

        # Flatten outputs for loss calculation.
        rel_logits = [e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)]
        # pad_size = 512 - rel_logits.size(-1)
        # if pad_size > 0:
        #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

        # for i, i_rel_logits in enumerate(rel_logits):
        #     pad_size = 512 - i_rel_logits.size(-1)
        #     if pad_size > 0:
        #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
        #     rel_logits[i] = i_rel_logits

        # num_pos = (edge_lbls != 0).sum()
        # num_neg = edge_lbls.numel() - num_pos
        # num_pos = get_ddp_average(num_pos, rel_logits[0].device)
        # num_neg = get_ddp_average(num_neg, rel_logits[0].device)

        rel_tgt_onehot = []
        for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
            i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

            if self.sgg_mode == 'full':
                # i_rel_tgt_onehot[i_edge_lbl] = 1.0
                i_rel_tgt_onehot[torch.arange(len(i_edge_lbl)), i_edge_lbl] = 1.0
            else:
                for ii, label in enumerate(i_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        rel_logits = torch.cat(rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)
        # rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        # #rel_num = rel_tgt_onehot.sum()
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(rel_num)
        # rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()
        rel_num = get_ddp_average(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device)

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 

        rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
        rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

        rel_mask = (rel_logits != float('-inf')).float()
        
        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()


        loss = (pos_loss + neg_loss) / rel_num
        losses = dict(loss_rln_labels=loss)
        # with torch.no_grad():
        #     losses['loss_rln_labels_pos'] = pos_loss.detach() /  rel_num
        #     losses['loss_rln_labels_neg'] = neg_loss.detach() /  rel_num

        losses['loss_rln_labels_pos'] = pos_loss / rel_num
        # losses['loss_rln_labels_pos'] = pos_loss / num_pos
        # losses['loss_rln_labels_neg'] = neg_loss / num_neg
        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

        return losses

    def loss_rln_distill(self, rln_hs, t_rln_hs, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls):
        # x_list = [x[i, :lengths[i], :] for i in range(b)]
        b, _, _ = t_rln_hs.shape
        t_rln_hs = [t_rln_hs[i, :batch_pair_num[i], :] for i in range(b)]
        # t_rln_hs = [x[permute] for x, permute in zip(t_rln_hs, random_permute_list)]

        rln_hs = [rln_hs[i, :batch_pair_num[i], :] for i in range(b)]

        t_rln_hs = torch.cat(t_rln_hs, dim=0)
        rln_hs = torch.cat(rln_hs, dim=0)
        edge_gt_lbls = torch.cat(edge_gt_lbls, dim=0)
        t_edge_gt_lbls = torch.cat(t_edge_gt_lbls, dim=0)

        assert torch.equal(edge_gt_lbls, t_edge_gt_lbls)

        rel_num = torch.tensor(rln_hs.shape[0]).to(t_rln_hs.device)
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        ids = torch.where(edge_gt_lbls == 0)[0]
        loss_distill = F.l1_loss(rln_hs[ids], t_rln_hs[ids], reduction='sum') / rel_num

        losses = dict(loss_distill=loss_distill)

        return losses

    def loss_encoder_distill(self, memory, memory_text, memory_t, memory_text_t, memory_mask, memory_text_mask):
        # get valid mask
        image_valid = (~memory_mask).float()
        text_valid = (~memory_text_mask).float()

        image_loss = F.l1_loss(memory, memory_t, reduction='none').mean(dim=-1)  # (bs, num_tokens)
        text_loss = F.l1_loss(memory_text, memory_text_t, reduction='none').mean(dim=-1)  # (bs, text_len)

        encoder_image_distill_loss = (image_loss * image_valid).sum() / image_valid.sum().clamp(min=1)
        encoder_text_distill_loss = (text_loss * text_valid).sum() / text_valid.sum().clamp(min=1)

        losses = dict(loss_encoder_distill=(encoder_image_distill_loss + encoder_text_distill_loss))
        return losses

    def loss_rln_logits_distill(self, rln_pred_logits, rln_pred_logits_t, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict):
        # Create input_id mask
        masks = []
        max_text_length = rln_pred_logits[0].shape[1]
        for i, i_input_ids in enumerate(rln_text_dict['input_ids']):
            mask = torch.zeros((batch_pair_num[i], max_text_length), dtype=torch.bool)
            mask[:, :len(i_input_ids)] = 1
            masks.append(mask)
        mask = torch.cat(masks, dim=0).to(rln_pred_logits[0].device)

        # Flatten logits and get rid of paddings.
        rel_logits = torch.cat([e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)])  # (n_q, text_length)
        rel_logits_t = torch.cat([e[:x] for e, x in zip(rln_pred_logits_t, batch_pair_num)])  # (n_q, text_length)

        edge_gt_lbls = torch.cat(edge_gt_lbls, dim=0)
        t_edge_gt_lbls = torch.cat(t_edge_gt_lbls, dim=0)
        assert torch.equal(edge_gt_lbls, t_edge_gt_lbls)
        valid_ids = torch.where(edge_gt_lbls == 0)[0]

        rel_logits = rel_logits[valid_ids]
        rel_logits_t = rel_logits_t[valid_ids]
        mask = mask[valid_ids]

        rel_logits = rel_logits[mask]
        rel_logits_t = rel_logits_t[mask]

        loss_sum = F.mse_loss(rel_logits, rel_logits_t, reduction='sum')
        # loss_sum = F.binary_cross_entropy_with_logits(rel_logits, torch.sigmoid(rel_logits_t).detach(), reduction='sum')

        numel = torch.tensor(rel_logits.numel(), device=rel_logits.device, dtype=torch.float32)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(loss_sum)
            torch.distributed.all_reduce(numel)
        # rel_num = torch.tensor(rel_logits.shape[0]).to(rel_logits.device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(rel_num)
        # rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        # loss_distill = F.mse_loss(rel_logits[valid_ids], rel_logits_t[valid_ids], reduction='sum') / rel_num
        loss_distill = loss_sum / torch.clamp(numel, min=1.0)
        losses = dict(loss_distill=loss_distill)
        return losses

    # def loss_pair_pruning(self, pair_selector_loss):
    #     loss = pair_selector_loss
    #     loss_dict = {"loss_pair_pruning": loss}

    #     return loss_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


    def loss_pair_pruning(self, pair_selector_gt_labels, pair_selector_scores, batch_pair_num):
        device = pair_selector_scores.device
        b, n = pair_selector_scores.shape

        # Construct valid mask
        valid_mask = torch.zeros((b, n), dtype=torch.bool, device=device)
        for bi in range(b):
            valid_mask[bi, :batch_pair_num[bi]] = True

        # Only compute on valid positions — avoids -inf → NaN
        logits = pair_selector_scores[valid_mask]          # (total_valid,)
        targets = pair_selector_gt_labels.float()[valid_mask]  # (total_valid,)

        # Focal loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha = 0.25
        gamma = 2.0
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss

        loss = focal_loss.mean()
        loss_dict = {'loss_pair_pruning': loss}
        return loss_dict

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        if 'input_ids' not in targets[0].keys():
            input_ids = outputs['text_dict']['input_ids']
            for bid, target in enumerate(targets):
                target['input_ids'] = input_ids[bid]
        indices = self.matcher(outputs_without_aux, targets)
        
        # if outputs_t is not None:
        #     outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
        #     indices_t = self.matcher(outputs_without_aux_t, targets)
        # else:
        #     indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # =============== dn loss ===============
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        # =============== standard loss ===============
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # =============== aux loss ===============
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_output in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_output, targets)
                # if outputs_t is not None:
                #     aux_outputs_t = outputs_t['aux_outputs'][idx]
                #     indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # =============== interm loss ===============
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            # if outputs_t is not None:
            #     interm_outputs_t = outputs_t['interm_outputs']
            #     indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # =============== enc_outputs loss ===============
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # ============================================= rln_decoder loss =============================================
        rln_out = outputs['rln_out']
        rln_text_dict = outputs['rel_text_dict']
        rln_query_mask = rln_out['rln_query_mask']
        batch_pair_num = rln_out['batch_pair_num']
        edge_gt_lbls_padded = outputs['edge_gt_lbls']
        edge_gt_boxes = outputs['edge_gt_boxes']

        rln_mask = (edge_gt_lbls_padded == 0)
        edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls_padded, batch_pair_num)]
        if outputs_t is not None:
            t_edge_gt_lbls_padded = outputs_t['edge_gt_lbls']
            t_edge_gt_lbls = [e[:x] for e, x in zip(t_edge_gt_lbls_padded, batch_pair_num)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_rln_boxes = sum(len(t["edges"]) for t in targets)
        num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_rln_boxes)
        num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

        # pair pruning loss
        # pair_pruning_loss (computed conditionally below)
        # pair_selector loss (backward compatible, only when pair_selector data is present)
        if 'pair_selector_scores' in rln_out and rln_out['pair_selector_scores'] is not None:
            pair_pruning_loss = self.loss_pair_pruning(
                rln_out['pair_selector_gt_labels'],
                rln_out['pair_selector_scores'],
                rln_out['batch_pair_num']
            )
            losses.update(pair_pruning_loss)
        # losses.update(pair_pruning_loss)

        # =============== interm class/box ===============
        interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
        interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
        losses.update(interm_loss_rln_boxes_dict)
        losses.update(interm_loss_rln_labels_dict)

        # =============== aux class/box ===============
        for idx, aux_output in enumerate(rln_out['aux_outputs']):
            aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
            aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
            aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
            aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
            losses.update(aux_loss_rln_boxes_dict)
            losses.update(aux_loss_rln_labels_dict)

        # =============== standard class / box ===============
        loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        losses.update(loss_rln_boxes_dict)
        losses.update(loss_rln_labels_dict)

        # =============== distill ===============
        if outputs_t is not None:
            if self.use_prob_distill:
                # interm
                interm_pred_logits = outputs['rln_out']['interm_outputs']['rln_pred_logits']
                interm_pred_logits_t = outputs_t['rln_out']['interm_outputs']['rln_pred_logits']
                interm_loss_distill_dict = self.loss_rln_logits_distill(interm_pred_logits, interm_pred_logits_t, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict)
                losses.update(interm_loss_distill_dict)

                # aux
                aux_outputs = outputs['rln_out']['aux_outputs']
                aux_outputs_t = outputs_t['rln_out']['aux_outputs']
                for idx, (aux_output, aux_output_t) in enumerate(zip(aux_outputs, aux_outputs_t)):
                    aux_loss_distill_dict = self.loss_rln_logits_distill(aux_output['rln_pred_logits'], aux_output_t['rln_pred_logits'], batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict)
                    aux_loss_distill_dict = {k + f'_{idx}': v for k, v in aux_loss_distill_dict.items()}
                    losses.update(aux_loss_distill_dict)

                # standard
                loss_distill_dict = self.loss_rln_logits_distill(outputs['rln_out']['rln_pred_logits'], outputs_t['rln_out']['rln_pred_logits'], batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict)
                losses.update(loss_distill_dict)
            else:
                t_interm_rln_hs = outputs_t['rln_out']['interm_outputs']['rln_hs']
                interm_rln_hs = outputs['rln_out']['interm_outputs']['rln_hs']
                interm_loss_distill_dict = self.loss_rln_distill(interm_rln_hs, t_interm_rln_hs, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls)
                losses.update(interm_loss_distill_dict)

                t_aux_outputs = outputs_t['rln_out']['aux_outputs']
                aux_outputs = outputs['rln_out']['aux_outputs']
                for idx, (t_aux_output, aux_output) in enumerate(zip(t_aux_outputs, aux_outputs)):
                    aux_loss_distill_dict = self.loss_rln_distill(aux_output['rln_hs'], t_aux_output['rln_hs'], batch_pair_num, edge_gt_lbls, t_edge_gt_lbls)
                    aux_loss_distill_dict = {k + f'_{idx}': v for k, v in aux_loss_distill_dict.items()}
                    losses.update(aux_loss_distill_dict)

                t_rln_hs = outputs_t['rln_out']['rln_hs']
                rln_hs = rln_out['rln_hs']
                loss_distill_dict = self.loss_rln_distill(rln_hs, t_rln_hs, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls)
                losses.update(loss_distill_dict)

            if 'loss_encoder_distill' in self.weight_dict:
                loss_encoder_distill_dict = self.loss_encoder_distill(
                    outputs['encoder_output']['memory'],
                    outputs['encoder_output']['memory_text'],
                    outputs_t['encoder_output']['memory'],
                    outputs_t['encoder_output']['memory_text'],
                    outputs['encoder_output']['memory_mask'],
                    outputs['encoder_output']['memory_text_mask']
                )
                losses.update(loss_encoder_distill_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups



class OvSGTRRelDecoderPairPruningCriterionV7(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,  matcher, weight_dict, focal_alpha, losses, 
                **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj=-hidden_dim*math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)

        self.use_prob_distill = kwargs.get('use_prob_distill', False)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'] # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']  # a list of gt box's name
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name +'.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])


        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)


        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, 
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}


        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)


        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        #with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return  {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def loss_rln_boxes(self, rln_pred_boxes, rln_gt_boxes, rln_mask, num_rln_boxes):
        """
        rln_pred_boxes: (bs, n, 4)
        rln_gt_boxes:   (bs, n, 4)
        rln_mask: (bs, n)  True = padding / invalid
        """
        # flatten
        src_boxes = rln_pred_boxes.reshape(-1, 4)
        target_boxes = rln_gt_boxes.reshape(-1, 4)
        valid_mask_flat = rln_mask.reshape(-1)  # (bs*n,)

        # keep only the valid boxes
        src_boxes = src_boxes[~valid_mask_flat]
        target_boxes = target_boxes[~valid_mask_flat]

        # guard against an all-padding batch (edge case)
        if src_boxes.numel() == 0:
            # return {
            #     'loss_rln_bbox': torch.tensor(0.0, device=rln_pred_boxes.device),
            #     'loss_rln_giou': torch.tensor(0.0, device=rln_pred_boxes.device),
            # }
            dummy_loss = rln_pred_boxes.sum() * 0.0
            return {
                'loss_rln_bbox': dummy_loss,
                'loss_rln_giou': dummy_loss,
            }

        # ---------------- L1 ----------------
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_rln_boxes

        # ---------------- GIoU ----------------
        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

        giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)

        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.sum() / num_rln_boxes

        return {
            'loss_rln_bbox': loss_bbox,
            'loss_rln_giou': loss_giou,
        }

    def loss_rln_labels(self, rln_pred_logits, edge_lbls, batch_pair_num, rel_text_dict):
        """
          compute loss for relations 
        """
        def get_ddp_average(num: int, device):
            num = torch.as_tensor(num, device=device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num)
            num = torch.clamp(num / get_world_size(), min=1).item()
            return num

        # Flatten outputs for loss calculation.
        rel_logits = [e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)]
        # pad_size = 512 - rel_logits.size(-1)
        # if pad_size > 0:
        #     rel_logits = F.pad(rel_logits, (0, pad_size), value=float('-inf'))

        # for i, i_rel_logits in enumerate(rel_logits):
        #     pad_size = 512 - i_rel_logits.size(-1)
        #     if pad_size > 0:
        #         i_rel_logits = F.pad(i_rel_logits, (0, pad_size), value=float('-inf'))
        #     rel_logits[i] = i_rel_logits

        # num_pos = (edge_lbls != 0).sum()
        # num_neg = edge_lbls.numel() - num_pos
        # num_pos = get_ddp_average(num_pos, rel_logits[0].device)
        # num_neg = get_ddp_average(num_neg, rel_logits[0].device)

        rel_tgt_onehot = []
        for i_rel_logits, i_edge_lbl, i_input_ids in zip(rel_logits, edge_lbls, rel_text_dict['input_ids']):
            i_rel_tgt_onehot = torch.zeros_like(i_rel_logits)

            if self.sgg_mode == 'full':
                # i_rel_tgt_onehot[i_edge_lbl] = 1.0
                i_rel_tgt_onehot[torch.arange(len(i_edge_lbl)), i_edge_lbl] = 1.0
            else:
                for ii, label in enumerate(i_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]':
                        continue 

                    ids = self.tokenizer(name +'.').input_ids[1:-1]
                    start_i, end_i = search_query_pos(i_input_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    i_rel_tgt_onehot[ii, start_i: end_i] = 1.0
            rel_tgt_onehot.append(i_rel_tgt_onehot)

        rel_logits = torch.cat(rel_logits)
        rel_tgt_onehot = torch.cat(rel_tgt_onehot)
        # rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device) 
        # #rel_num = rel_tgt_onehot.sum()
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(rel_num)
        # rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()
        rel_num = get_ddp_average(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device)

        alpha, gamma = 0.25, 2.0
        eps = 1e-5 

        rel_logits = rel_logits.float()  # This is added because sometimes prob may pass 1.0 due to dtype conversion, not OvSGTR original impliment.
        rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0-eps)

        rel_mask = (rel_logits != float('-inf')).float()
        
        rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
        rel_weight[rel_weight == 0] = 1.0

        pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob)** gamma) * rel_tgt_onehot * rel_mask / rel_weight
        neg_loss = - torch.log(1.0 -rel_prob) * (rel_prob**gamma) * (1.0 - rel_tgt_onehot) * rel_mask  / rel_weight

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()


        loss = (pos_loss + neg_loss) / rel_num
        losses = dict(loss_rln_labels=loss)
        # with torch.no_grad():
        #     losses['loss_rln_labels_pos'] = pos_loss.detach() /  rel_num
        #     losses['loss_rln_labels_neg'] = neg_loss.detach() /  rel_num

        losses['loss_rln_labels_pos'] = pos_loss / rel_num
        # losses['loss_rln_labels_pos'] = pos_loss / num_pos
        # losses['loss_rln_labels_neg'] = neg_loss / num_neg
        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_rln_labels'].device)

        return losses

    def loss_rln_distill(self, rln_hs, t_rln_hs, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls):
        # x_list = [x[i, :lengths[i], :] for i in range(b)]
        b, _, _ = t_rln_hs.shape
        t_rln_hs = [t_rln_hs[i, :batch_pair_num[i], :] for i in range(b)]
        # t_rln_hs = [x[permute] for x, permute in zip(t_rln_hs, random_permute_list)]

        rln_hs = [rln_hs[i, :batch_pair_num[i], :] for i in range(b)]

        t_rln_hs = torch.cat(t_rln_hs, dim=0)
        rln_hs = torch.cat(rln_hs, dim=0)
        edge_gt_lbls = torch.cat(edge_gt_lbls, dim=0)
        t_edge_gt_lbls = torch.cat(t_edge_gt_lbls, dim=0)

        assert torch.equal(edge_gt_lbls, t_edge_gt_lbls)

        rel_num = torch.tensor(rln_hs.shape[0]).to(t_rln_hs.device)
        #rel_num = rel_tgt_onehot.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        ids = torch.where(edge_gt_lbls == 0)[0]
        loss_distill = F.l1_loss(rln_hs[ids], t_rln_hs[ids], reduction='sum') / rel_num

        losses = dict(loss_distill=loss_distill)

        return losses

    def loss_encoder_distill(self, memory, memory_text, memory_t, memory_text_t, memory_mask, memory_text_mask):
        # get valid mask
        image_valid = (~memory_mask).float()
        text_valid = (~memory_text_mask).float()

        image_loss = F.l1_loss(memory, memory_t, reduction='none').mean(dim=-1)  # (bs, num_tokens)
        text_loss = F.l1_loss(memory_text, memory_text_t, reduction='none').mean(dim=-1)  # (bs, text_len)

        encoder_image_distill_loss = (image_loss * image_valid).sum() / image_valid.sum().clamp(min=1)
        encoder_text_distill_loss = (text_loss * text_valid).sum() / text_valid.sum().clamp(min=1)

        losses = dict(loss_encoder_distill=(encoder_image_distill_loss + encoder_text_distill_loss))
        return losses

    def loss_rln_logits_distill(self, rln_pred_logits, rln_pred_logits_t, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict):
        # Create input_id mask
        masks = []
        max_text_length = rln_pred_logits[0].shape[1]
        for i, i_input_ids in enumerate(rln_text_dict['input_ids']):
            mask = torch.zeros((batch_pair_num[i], max_text_length), dtype=torch.bool)
            mask[:, :len(i_input_ids)] = 1
            masks.append(mask)
        mask = torch.cat(masks, dim=0).to(rln_pred_logits[0].device)

        # Flatten logits and get rid of paddings.
        rel_logits = torch.cat([e[:x] for e, x in zip(rln_pred_logits, batch_pair_num)])  # (n_q, text_length)
        rel_logits_t = torch.cat([e[:x] for e, x in zip(rln_pred_logits_t, batch_pair_num)])  # (n_q, text_length)

        edge_gt_lbls = torch.cat(edge_gt_lbls, dim=0)
        t_edge_gt_lbls = torch.cat(t_edge_gt_lbls, dim=0)
        assert torch.equal(edge_gt_lbls, t_edge_gt_lbls)
        valid_ids = torch.where(edge_gt_lbls == 0)[0]

        rel_logits = rel_logits[valid_ids]
        rel_logits_t = rel_logits_t[valid_ids]
        mask = mask[valid_ids]

        rel_logits = rel_logits[mask]
        rel_logits_t = rel_logits_t[mask]

        loss_sum = F.mse_loss(rel_logits, rel_logits_t, reduction='sum')
        # loss_sum = F.binary_cross_entropy_with_logits(rel_logits, torch.sigmoid(rel_logits_t).detach(), reduction='sum')

        numel = torch.tensor(rel_logits.numel(), device=rel_logits.device, dtype=torch.float32)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(loss_sum)
            torch.distributed.all_reduce(numel)
        # rel_num = torch.tensor(rel_logits.shape[0]).to(rel_logits.device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(rel_num)
        # rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        # loss_distill = F.mse_loss(rel_logits[valid_ids], rel_logits_t[valid_ids], reduction='sum') / rel_num
        loss_distill = loss_sum / torch.clamp(numel, min=1.0)
        losses = dict(loss_distill=loss_distill)
        return losses

    # def loss_pair_pruning(self, pair_selector_loss):
    #     loss = pair_selector_loss
    #     loss_dict = {"loss_pair_pruning": loss}

    #     return loss_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp*pred_obj)

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0-eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0-prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


    def loss_pair_pruning(self, pair_selector_gt_labels, pair_selector_scores, batch_pair_num):
        device = pair_selector_scores.device
        b, n = pair_selector_scores.shape

        # Construct valid mask
        valid_mask = torch.zeros((b, n), dtype=torch.bool, device=device)
        for bi in range(b):
            valid_mask[bi, :batch_pair_num[bi]] = True

        # Only compute on valid positions — avoids -inf → NaN
        logits = pair_selector_scores[valid_mask]          # (total_valid,)
        targets = pair_selector_gt_labels.float()[valid_mask]  # (total_valid,)

        # Focal loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha = 0.25
        gamma = 2.0
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss

        loss = focal_loss.mean()
        loss_dict = {'loss_pair_pruning': loss}
        return loss_dict

    def forward(self, outputs, targets, outputs_t=None, return_indices=False, 
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        self.global_iter = global_iter

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device

        # indices indicates gt box and pred box matching index. (B, 2, num_gt_box). (0, 0, 0) <-> (0, 1, 0) query match with gt.
        if 'input_ids' not in targets[0].keys():
            input_ids = outputs['text_dict']['input_ids']
            for bid, target in enumerate(targets):
                target['input_ids'] = input_ids[bid]
        indices = self.matcher(outputs_without_aux, targets)
        
        # if outputs_t is not None:
        #     outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
        #     indices_t = self.matcher(outputs_without_aux_t, targets)
        # else:
        #     indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # =============== dn loss ===============
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None 

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        # =============== standard loss ===============
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # =============== aux loss ===============
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_output in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_output, targets)
                # if outputs_t is not None:
                #     aux_outputs_t = outputs_t['aux_outputs'][idx]
                #     indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_output, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # =============== interm loss ===============
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            # if outputs_t is not None:
            #     interm_outputs_t = outputs_t['interm_outputs']
            #     indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue 
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # =============== enc_outputs loss ===============
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # ============================================= rln_decoder loss =============================================
        rln_out = outputs['rln_out']
        rln_text_dict = outputs['rel_text_dict']
        rln_query_mask = rln_out['rln_query_mask']
        batch_pair_num = rln_out['batch_pair_num']
        edge_gt_lbls_padded = outputs['edge_gt_lbls']
        edge_gt_boxes = outputs['edge_gt_boxes']

        rln_mask = (edge_gt_lbls_padded == 0)
        edge_gt_lbls = [e[:x] for e, x in zip(edge_gt_lbls_padded, batch_pair_num)]
        if outputs_t is not None:
            t_edge_gt_lbls_padded = outputs_t['edge_gt_lbls']
            t_edge_gt_lbls = [e[:x] for e, x in zip(t_edge_gt_lbls_padded, batch_pair_num)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_rln_boxes = sum(len(t["edges"]) for t in targets)
        num_rln_boxes = torch.as_tensor([num_rln_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_rln_boxes)
        num_rln_boxes = torch.clamp(num_rln_boxes / get_world_size(), min=1).item()

        # pair pruning loss
        # pair_pruning_loss (computed conditionally below)
        # pair_selector loss (backward compatible, only when pair_selector data is present)
        if 'pair_selector_scores' in rln_out and rln_out['pair_selector_scores'] is not None:
            pair_pruning_loss = self.loss_pair_pruning(
                rln_out['pair_selector_gt_labels'],
                rln_out['pair_selector_scores'],
                rln_out['batch_pair_num']
            )
            losses.update(pair_pruning_loss)
        # losses.update(pair_pruning_loss)

        # =============== interm class/box ===============
        interm_loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['interm_outputs']['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        interm_loss_rln_labels_dict = self.loss_rln_labels(rln_out['interm_outputs']['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        interm_loss_rln_boxes_dict = {k + f'_interm': v for k, v in interm_loss_rln_boxes_dict.items()}
        interm_loss_rln_labels_dict = {k + f'_interm': v for k, v in interm_loss_rln_labels_dict.items()}
        losses.update(interm_loss_rln_boxes_dict)
        losses.update(interm_loss_rln_labels_dict)

        # =============== aux class/box ===============
        for idx, aux_output in enumerate(rln_out['aux_outputs']):
            aux_loss_rln_boxes_dict = self.loss_rln_boxes(aux_output['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
            aux_loss_rln_labels_dict = self.loss_rln_labels(aux_output['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
            aux_loss_rln_boxes_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_boxes_dict.items()}
            aux_loss_rln_labels_dict = {k + f'_{idx}': v for k, v in aux_loss_rln_labels_dict.items()}
            losses.update(aux_loss_rln_boxes_dict)
            losses.update(aux_loss_rln_labels_dict)

        # =============== standard class / box ===============
        loss_rln_boxes_dict = self.loss_rln_boxes(rln_out['rln_pred_boxes'], edge_gt_boxes, rln_mask, num_rln_boxes)
        loss_rln_labels_dict = self.loss_rln_labels(rln_out['rln_pred_logits'], edge_gt_lbls, batch_pair_num, rln_text_dict)
        losses.update(loss_rln_boxes_dict)
        losses.update(loss_rln_labels_dict)

        # =============== distill ===============
        if outputs_t is not None:
            if self.use_prob_distill:
                # interm
                interm_pred_logits = outputs['rln_out']['interm_outputs']['rln_pred_logits']
                interm_pred_logits_t = outputs_t['rln_out']['interm_outputs']['rln_pred_logits']
                interm_loss_distill_dict = self.loss_rln_logits_distill(interm_pred_logits, interm_pred_logits_t, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict)
                losses.update(interm_loss_distill_dict)

                # aux
                aux_outputs = outputs['rln_out']['aux_outputs']
                aux_outputs_t = outputs_t['rln_out']['aux_outputs']
                for idx, (aux_output, aux_output_t) in enumerate(zip(aux_outputs, aux_outputs_t)):
                    aux_loss_distill_dict = self.loss_rln_logits_distill(aux_output['rln_pred_logits'], aux_output_t['rln_pred_logits'], batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict)
                    aux_loss_distill_dict = {k + f'_{idx}': v for k, v in aux_loss_distill_dict.items()}
                    losses.update(aux_loss_distill_dict)

                # standard
                loss_distill_dict = self.loss_rln_logits_distill(outputs['rln_out']['rln_pred_logits'], outputs_t['rln_out']['rln_pred_logits'], batch_pair_num, edge_gt_lbls, t_edge_gt_lbls, rln_text_dict)
                losses.update(loss_distill_dict)
            else:
                t_interm_rln_hs = outputs_t['rln_out']['interm_outputs']['rln_hs']
                interm_rln_hs = outputs['rln_out']['interm_outputs']['rln_hs']
                interm_loss_distill_dict = self.loss_rln_distill(interm_rln_hs, t_interm_rln_hs, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls)
                losses.update(interm_loss_distill_dict)

                t_aux_outputs = outputs_t['rln_out']['aux_outputs']
                aux_outputs = outputs['rln_out']['aux_outputs']
                for idx, (t_aux_output, aux_output) in enumerate(zip(t_aux_outputs, aux_outputs)):
                    aux_loss_distill_dict = self.loss_rln_distill(aux_output['rln_hs'], t_aux_output['rln_hs'], batch_pair_num, edge_gt_lbls, t_edge_gt_lbls)
                    aux_loss_distill_dict = {k + f'_{idx}': v for k, v in aux_loss_distill_dict.items()}
                    losses.update(aux_loss_distill_dict)

                t_rln_hs = outputs_t['rln_out']['rln_hs']
                rln_hs = rln_out['rln_hs']
                loss_distill_dict = self.loss_rln_distill(rln_hs, t_rln_hs, batch_pair_num, edge_gt_lbls, t_edge_gt_lbls)
                losses.update(loss_distill_dict)

            if 'loss_encoder_distill' in self.weight_dict:
                loss_encoder_distill_dict = self.loss_encoder_distill(
                    outputs['encoder_output']['memory'],
                    outputs['encoder_output']['memory_text'],
                    outputs_t['encoder_output']['memory'],
                    outputs_t['encoder_output']['memory_text'],
                    outputs['encoder_output']['memory_mask'],
                    outputs['encoder_output']['memory_text_mask']
                )
                losses.update(loss_encoder_distill_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


