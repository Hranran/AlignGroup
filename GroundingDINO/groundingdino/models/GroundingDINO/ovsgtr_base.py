# ------------------------------------------------------------------------
# modified from Grounding DINO
# url: https://github.com/gpt4vision/OvSGTR
# Copyright (c) 2024 Joseph Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# 
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
'''
This model is a replicate of origin OvSGTR, but move every loss compute procedure
back to the model forward itself.
'''
import os
import copy
from typing import List, Dict
import random
import numpy as np

import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import batched_nms

import torch.distributed as dist

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
    all_gather 
)

from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span

from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn


from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, gen_sineembed_for_position

from .utils import FrequencyBias

# 
import bisect
from .matcher import build_matcher 
from .losses import OvSGTRBaseCriterion
from .dn_components import prepare_for_cdn,dn_post_process

MAX_WORDS_LEN = 151


def is_main_process():
    return (not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x


class OvSGTRBase(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""
    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        hidden_dim=256,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        num_classes=2000,
        # sgg
        sgg_mode="full",
        num_rln_cat=51,
        num_rln_queries=1,
        rln_freq_bias=None,
        focal_loss_for_edges=False,
        concat_rel_text=True,
        rel_batch_per_image=64,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        self.label_enc = nn.Embedding(dn_labelbook_size+1, hidden_dim)

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)


        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed(max_text_len=self.max_text_len) # 2048

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed


        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.refpoint_embed = None

        # ====================== sgg part ======================
        self.sgg_mode = sgg_mode
        self.focal_loss_for_edges = focal_loss_for_edges
        self.concat_rel_text = concat_rel_text
        self.rel_batch_per_image = rel_batch_per_image

        self.num_rln_queries = num_rln_queries
        if self.num_rln_queries > 0:
            self.rln_embed = nn.Embedding(self.num_rln_queries, self.hidden_dim*2)
            self.transformer.decoder.rln_embed = self.rln_embed
        else:
            self.transformer.decoder.rln_embed = None

        input_dim = self.hidden_dim * (2 + self.num_rln_queries)

        self.rln_text_proj = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        self.rln_proj = MLP(input_dim, 2048, self.hidden_dim, 2) # 2048->1024

        if self.sgg_mode == 'full':
            self.rln_classifier = nn.Linear(self.hidden_dim, num_rln_cat)
            self.rln_freq_bias = FrequencyBias(rln_freq_bias) if rln_freq_bias else None
            if rln_freq_bias:
                nn.init.constant_(self.rln_classifier.bias, 0.)
            elif focal_loss_for_edges:
                prior_prob = 0.1
                bias_value = -math.log((1 - prior_prob) / prior_prob)

                self.rln_classifier.weight.data.fill_(0.) 
                nn.init.constant_(self.rln_classifier.bias, bias_value)
        else:
            self.rln_freq_bias = None
            self.rln_classifier = None 

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def encode_captions(self, captions, device, encode_relation=False):
        text_dict = {
            "encoded_text": [], #encoded_text,  # bs, 195, d_model
            "text_token_mask": [], #text_token_mask,  # bs, 195
            "position_ids": [], #position_ids,  # bs, 195
            "text_self_attention_masks": [], #text_self_attention_masks,  # bs, 195,195
            "input_ids": []
        }
        cap_len = [len(e[:-1].split('.')) for e in captions]
        captions = [e[:-1].split('.') for e in captions]

        for ii in range((max(cap_len) + MAX_WORDS_LEN-1) // MAX_WORDS_LEN):
            sub_captions = [e[ii*MAX_WORDS_LEN: (ii+1)*MAX_WORDS_LEN] for e in captions]
            sub_captions = [('. '.join(e) + '. ') for e in sub_captions]
            
            # encoder texts
            tokenized = self.tokenizer(sub_captions, padding="longest", return_tensors="pt").to(
                device
            )

            (
                text_self_attention_masks,  # (bs, text_length, text_length)
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, self.tokenizer
            )

            if text_self_attention_masks.shape[1] > self.max_text_len:
                print("Warning: text_self_attention_masks.shape[1]:{} > max_text_len:{}".format(
                      text_self_attention_masks.shape[1], self.max_text_len))

                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]
                position_ids = position_ids[:, : self.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

            # extract text embeddings
            if self.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                tokenized_for_encoder = tokenized

            bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

            if encode_relation:
                assert self.rln_text_proj is not None, "rln_text_proj cannot be None !"
                encoded_text = self.rln_text_proj(bert_output["last_hidden_state"])  # bs, 195, d_model
            else:
                encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model


            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask

            if encoded_text.shape[1] > self.max_text_len:
                print("Warning: encoded_text.shape[1]:{} > max_text_len:{}".format(encoded_text.shape[1],
                        self.max_text_len))
                encoded_text = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[
                    :, : self.max_text_len, : self.max_text_len
                ]

            text_dict['encoded_text'].append(encoded_text)
            text_dict['text_token_mask'].append(text_token_mask)
            text_dict['position_ids'].append(position_ids)
            text_dict['text_self_attention_masks'].append(text_self_attention_masks)
            text_dict['input_ids'].append(tokenized.input_ids)

        text_dict['encoded_text'] = torch.cat(text_dict['encoded_text'], 1)
        text_dict['text_token_mask'] = torch.cat(text_dict['text_token_mask'], 1)
        text_dict['position_ids'] = torch.cat(text_dict['position_ids'], 1)
        text_dict['input_ids'] = torch.cat(text_dict['input_ids'], 1)
        attn_mask = torch.zeros((text_dict['encoded_text'].shape[0], 
                                 text_dict['encoded_text'].shape[1],
                                 text_dict['encoded_text'].shape[1]), 
                                 device=device, dtype=torch.bool)

        last_idx = 0
        for ii, mask in enumerate(text_dict['text_self_attention_masks']):
            attn_mask[:, last_idx: last_idx + mask.shape[1], last_idx: last_idx+mask.shape[1]] = mask
            last_idx += mask.shape[1]

        text_dict['text_self_attention_masks'] = attn_mask
        return text_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def forward(self, samples: NestedTensor, targets: List = None, permute_idx=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        captions = [t["caption"] for t in targets]
        rel_captions = [t['rel_caption'] for t in targets]

        # text features
        text_dict = self.encode_captions(captions, samples.device, encode_relation=False)
        rel_text_dict = self.encode_captions(rel_captions, samples.device, encode_relation=True)

        if self.concat_rel_text:
            sep_len = (text_dict['encoded_text'].shape[1], rel_text_dict['encoded_text'].shape[1])
            for k in text_dict.keys():
                if k != 'text_self_attention_masks':
                    text_dict[k] = torch.cat((text_dict[k], rel_text_dict[k]), 1)
            attn_mask = torch.zeros((text_dict['encoded_text'].shape[0], 
                                 text_dict['encoded_text'].shape[1],
                                 text_dict['encoded_text'].shape[1]), 
                                 device=samples.device, dtype=torch.bool)
            attn_mask[:, :sep_len[0], :sep_len[0]] = text_dict['text_self_attention_masks']
            attn_mask[:, sep_len[0]:, sep_len[0]:] = rel_text_dict['text_self_attention_masks']
            text_dict['text_self_attention_masks'] = attn_mask

        # visual features
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
            backbone_feature = None 


        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # dn part 
        use_dn = self.dn_number > 0 and targets is not None  # No, dn_number is 0
        if use_dn:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim, label_enc=self.label_enc)
        else:
            #assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None        

        hs, hs_rln, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, 
            input_query_label, attn_mask, text_dict,
        )

        # split via sep_len
        if self.concat_rel_text:
            rel_text_dict = {}
            for k in text_dict.keys():
                if k != 'text_self_attention_masks' and k != 'sep_len':
                    tmp = text_dict[k].split(sep_len, 1)
                    text_dict[k] = tmp[0]
                    rel_text_dict[k] = tmp[1]

            rel_text_dict['text_self_attention_masks'] = text_dict['text_self_attention_masks'][:, sep_len[0]:, sep_len[0]:]
            text_dict['text_self_attention_masks'] = text_dict['text_self_attention_masks'][:, :sep_len[0], :sep_len[0]]


        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        if self.dn_number > 0 and dn_meta is not None:
            hs_obj = [e[:, dn_meta['pad_size']:, :] for e in hs]
            outputs_class, outputs_coord_list = dn_post_process(outputs_class, outputs_coord_list, dn_meta, self.aux_loss, self._set_aux_loss)
        else:
            hs_obj = hs


        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord }
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal,}

        if dn_meta is not None:
            out['dn_meta'] = dn_meta

        # =============== sgg ===============
        out['input_ids'] = text_dict['input_ids'] #tokenized['input_ids']
        out['hs_obj'] = hs_obj[-1]
        out['hs_rln'] = hs_rln[-1] if hs_rln else None
        out['rel_text_dict'] = rel_text_dict

        if 'interm_outputs' in out:
            out['interm_outputs']['hs_obj'] = hs_obj[-1]
            out['interm_outputs']['hs_rln'] = hs_rln[-1] if hs_rln else None

        if self.aux_loss:
            for idx, item in enumerate(out['aux_outputs']):
                item['hs_obj'] = hs_obj[idx]
                item['hs_rln'] = hs_rln[idx] if hs_rln else None

        input_ids = out['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]

        # For each out, the following is required:
        # pred_logits, pred_boxes, input_ids: for GDINO loss
        # rel_text_dict, hs_obj: for sgg loss

        edge_lbls, relation_features, random_permute_list = self.sgg_forward(out, targets, permute_idx)
        out['edge_lbls'] = edge_lbls
        out['relation_features'] = relation_features

        if "interm_outputs" in out:
            edge_lbls, relation_features, _ = self.sgg_forward(out['interm_outputs'], targets, random_permute_list if permute_idx is None else permute_idx)
            out['interm_outputs']['edge_lbls'] = edge_lbls
            out['interm_outputs']['relation_features'] = relation_features

        if self.aux_loss:
            for idx, item in enumerate(out['aux_outputs']):
                edge_lbls, relation_features, _ = self.sgg_forward(item, targets, random_permute_list if permute_idx is None else permute_idx)
                item['edge_lbls'] = edge_lbls
                item['relation_features'] = relation_features

        return out, random_permute_list

    def sgg_forward(self, outputs, targets, permute_idx):
        """
        sample pairs.
        """
        indices = self.matcher(outputs, targets)
        object_token = outputs['hs_obj']
        relation_token = outputs['hs_rln']

        b, _, _ = object_token.shape

        permute_idx_list = [{} for _ in range(b)]
        if permute_idx is not None:
            permute_idx_list = permute_idx

        relation_features = []
        edge_lbls = []
        batch_pair_num = []
        for b, target in enumerate(targets):
            tgt_edges = target['edges']
            gt_to_query_mapping = {}
            for src, dst in zip(indices[b][0].tolist(), indices[b][1].tolist()):  # src: query index. dst: gt box index.
                gt_to_query_mapping[dst] = src

            # fetch gt edges.
            i_relation_feature_list = []
            i_all_edge_lbl = []
            for tgt_edge in tgt_edges:
                subj_feat = object_token[b, gt_to_query_mapping[tgt_edge[0].item()]]
                obj_feat  = object_token[b, gt_to_query_mapping[tgt_edge[1].item()]]
                if relation_token is not None:
                    rel_feat = relation_token[b].flatten()
                    feat = torch.cat([subj_feat, obj_feat, rel_feat])
                else:
                    feat = torch.cat([subj_feat, obj_feat])
                i_relation_feature_list.append(feat)
                # i_relation_feature_list.append(torch.cat([
                #     object_token[b, gt_to_query_mapping[tgt_edge[0].item()]],
                #     object_token[b, gt_to_query_mapping[tgt_edge[1].item()]],
                #     relation_token[b].flatten()
                #     ]))
                i_all_edge_lbl.append(tgt_edge[2].item())

            # fetch negative samples
            num_gt_box = len(gt_to_query_mapping)
            relation_matrix = torch.ones((num_gt_box, num_gt_box)) - torch.diag(torch.ones(num_gt_box))
            for tgt_edge in tgt_edges:
                relation_matrix[tgt_edge[0], tgt_edge[1]] = 0
            negative_sampled_edges = torch.nonzero(relation_matrix)
            if self.sgg_mode == 'full':
                num_neg_per_img = max(1, len(target['edges']) * 3)
            else:
                num_neg_per_img = max(1, self.rel_batch_per_image - len(target['edges']))

            if len(negative_sampled_edges) >= num_neg_per_img:
                # idx_ = torch.randperm(negative_sampled_edges.shape[0])[:num_neg_per_img]
                # negative_sampled_edges = negative_sampled_edges[idx_]
                if permute_idx is None:
                    idx_ = torch.randperm(negative_sampled_edges.shape[0])[:num_neg_per_img]
                    permute_idx_list[b]['negative_sampled_edges_permute'] = idx_
                else:
                    idx_ = permute_idx_list[b].get('negative_sampled_edges_permute', None)
                negative_sampled_edges = negative_sampled_edges[idx_]

            for neg_edge in negative_sampled_edges:
                subj_feat = object_token[b, gt_to_query_mapping[neg_edge[0].item()]]
                obj_feat  = object_token[b, gt_to_query_mapping[neg_edge[1].item()]]
                if relation_token is not None:
                    rel_feat = relation_token[b].flatten()
                    feat = torch.cat([subj_feat, obj_feat, rel_feat])
                else:
                    feat = torch.cat([subj_feat, obj_feat])
                i_relation_feature_list.append(feat)
                # i_relation_feature_list.append(torch.cat([
                #     object_token[b, gt_to_query_mapping[neg_edge[0].item()]],
                #     object_token[b, gt_to_query_mapping[neg_edge[1].item()]],
                #     relation_token[b].flatten()
                #     ]))
                i_all_edge_lbl.append(0)

            i_relation_feature_list = torch.stack(i_relation_feature_list)
            i_all_edge_lbl = torch.tensor(i_all_edge_lbl).to(object_token.device)
            assert len(i_relation_feature_list) > 0, "No relation features !"

            relation_features.append(i_relation_feature_list)
            edge_lbls.append(i_all_edge_lbl)
            batch_pair_num.append(len(i_relation_feature_list))

        # relation_feature: (bs, num_rln, d)
        # all_edge_lbl: (bs, num_rln)
        # random_permute_list = [torch.randperm(pair_num) for pair_num in batch_pair_num]

        random_permute_list = None
        if permute_idx is None:
            random_permute_list = [torch.randperm(pair_num) for pair_num in batch_pair_num]
            for i_random_permute, i_permute_idx_list in zip(random_permute_list, permute_idx_list):
                i_permute_idx_list['rln_random_permute'] = i_random_permute
        else:
            random_permute_list = [i_permute_idx_list['rln_random_permute'] for i_permute_idx_list in permute_idx_list]


        relation_features = [feats[idx] for feats, idx in zip(relation_features, random_permute_list)]
        edge_lbls = [e[idx] for e, idx in zip(edge_lbls, random_permute_list)]

        relation_features = torch.nn.utils.rnn.pad_sequence(relation_features, batch_first=True)
        relation_features = self.rln_proj(relation_features)
        relation_features = [feat[:num] for feat, num in zip(relation_features, batch_pair_num)]

        # edge_lbl: list
        # relation_featureL list
        return edge_lbls, relation_features, permute_idx_list


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1, 
                 do_sgg=False, relation_thresh=0.5,
                 test_overlap=False, use_gt_box=False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.do_sgg = do_sgg
        self.relation_thresh = relation_thresh
        self.test_overlap = test_overlap
        self.use_gt_box = use_gt_box 
        self.rln_proj = kwargs.get("rln_proj", None)
        self.detections_per_img = kwargs.get("detections_per_img", 100)
        self.temperature = kwargs.get("temperature", 1)

        self.tokenizer = kwargs.get("tokenizer", None)
        self.name2classes = kwargs.get("name2classes", None)
        self.name2predicates = kwargs.get("name2predicates", None)
        self.max_text_len = kwargs.get("max_text_len", 2048)
        self.matcher = kwargs.get("matcher", None)

        assert self.tokenizer is not None, " tokenzier should not be None when use text labels !"
        self.text_threshold = 1e-3
        self._positive_map = None

    def get_positive_map(self, max_text_len=2048, cat_list=None):
        if self._positive_map is not None:
            return self._positive_map

        if cat_list is None:
            assert self.name2classes is not None, "self.name2classes should not be None!"
            cat_list = [e for e in self.name2classes.keys()]
            use_name2class = True
        else:
            use_name2class = False 

        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
                    self.tokenizer(captions), tokenspanlist, max_text_len)  # N, dim. normed            
        
        if use_name2class:
            max_label = max(self.name2classes.values()) + 1
            new_pos_map = torch.zeros((max_label, positive_map.shape[1]))
            for ii, (name, label) in enumerate(self.name2classes.items()):
                new_pos_map[label] = positive_map[ii]
        else:
            new_pos_map = positive_map

        self._positive_map = new_pos_map  # (151, 512) for use_name2classes       
        return self._positive_map

    def __repr__(self):
        return f"{self.__class__.__name__}(num_select={self.num_select},\n\t  nms_iou_threshold={self.nms_iou_threshold},\n\t detections_per_img={self.detections_per_img},\n\t ,  relation_thresh={self.relation_thresh}, \n\t test_overlap={self.test_overlap}, \n\t use_gt_box={self.use_gt_box}, \n\t max_text_len={self.max_text_len})"

    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets, cat_list=None, gt_dicts=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select  # How many boxes saved for OBD part.
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        device = out_logits.device

        obj_token = outputs['hs_obj']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        batch_size = out_logits.shape[0]

        pos_maps = self.get_positive_map(self.max_text_len, cat_list).to(prob.device)  # For each category, find its' positive position in input_ids. (num_cat, text_length) THIS IS FOR OBD!!
        prob_to_label = prob @ pos_maps.T  # (bs, num_queries, text_length) * (num_cat, text_length)^T
        prob = prob_to_label  # (bs, num_queries, num_cat)

        num_cat = prob.shape[2]
        topk_values, topk_indexes = torch.topk(prob.view(batch_size, -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // num_cat
        labels = topk_indexes % num_cat

        out_bbox = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        boxes = out_bbox
        origin_output_boxes = copy.deepcopy(out_bbox)
        dim = obj_token.shape[-1]
        obj_token = torch.gather(obj_token, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, dim))

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if gt_dicts:
            boxes = gt_dicts[0]['gt_boxes'].unsqueeze(0)  # (b, num_box, 4)
            graph_infer_used_bbox = copy.deepcopy(boxes)
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            boxes = boxes * scale_fct[:, None, :]

            labels = gt_dicts[0]['gt_labels'].unsqueeze(0)
            scores = torch.ones_like(labels).float()

            src_ids, dst_ids = gt_dicts[0]['ids']  # match idx of query <-> gt 
            tmp = [-1] * len(dst_ids)
            for src_, dst_ in zip(src_ids.tolist(), dst_ids.tolist()):
                tmp[dst_] = src_
            tmp = torch.as_tensor(tmp)
            obj_token = outputs['hs_obj'][:, tmp]

        # nms if required
        if self.nms_iou_threshold > 0:
            item_indices = [batched_nms(b, s, l, iou_threshold=self.nms_iou_threshold) for b,s,l in zip(boxes, scores, labels)]
            item_indices = [e[:self.detections_per_img] for e in item_indices]
            results = [
                {
                    'scores': s[i],
                    'labels': l[i],
                    'boxes': b[i],
                    'obj_token': ot[i],
                    'origin_output_boxes': ob[i]
                    } for s, l, b, ot, ob, i in zip(scores, labels, boxes, obj_token, origin_output_boxes, item_indices)]
        else:
            results = [
                {
                    'scores': s[:self.detections_per_img],
                    'labels': l[:self.detections_per_img],
                    'boxes': b[:self.detections_per_img],
                    'obj_token': ot[:self.detections_per_img],
                    'origin_output_boxes': ob[:self.detections_per_img]
                    } for s, l, b, ot, ob in zip(scores, labels, boxes, obj_token, origin_output_boxes)]

        _ = [result.update({'scale_fct': scale_fct[batch_id]}) for batch_id, result in enumerate(results)]

        sgg_out = self.graph_infer(
            results,
            outputs['hs_rln'],
            self.rln_proj,
            outputs['rel_text_dict'],
            self.name2predicates,
            self.tokenizer,
            targets
            )

        for batch, res in enumerate(results):
            res.update({'graph': sgg_out[batch]})
            # del res['rln_token']
            del res['obj_token']

        return results

    def graph_infer(self,
                    outputs : List[Dict],
                    hs_rln,
                    rln_proj,
                    text_dict,
                    name2predicates, 
                    tokenizer,
                    targets
                    ):
        dst = []
        for batch_id, output in enumerate(outputs):
            origin_output_boxes = output['origin_output_boxes']

            obj_token = output['obj_token'] # (#obj, dim)
            relation_token = hs_rln[batch_id] if hs_rln is not None else None

            encoded_text = text_dict['encoded_text'][batch_id]
            text_mask = text_dict['text_token_mask'][batch_id]
            input_ids = text_dict['input_ids'][batch_id]
            sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]

            boxes = copy.deepcopy(output['boxes']) #(#obj, 4)
            scores = copy.deepcopy(output['scores']) # (#obj)
            labels = copy.deepcopy(output['labels']) # (#obj)

            node_id = torch.nonzero(labels).squeeze() 

            obj_token = obj_token[node_id]
            pred_classes = labels[node_id]
            pred_cls_score = scores[node_id]
            origin_output_boxes = origin_output_boxes[node_id]

            pred_boxes = boxes[node_id]
            pred_boxes_score = pred_cls_score
            pred_boxes_class = pred_classes


            if node_id.dim() !=0 and node_id.nelement() != 0 and node_id.shape[0]>1:
                # all possible node pairs in all token ordering
                tmp = torch.arange(len(node_id))
                node_pairs = torch.cat((torch.combinations(tmp), torch.combinations(tmp)[:, [1, 0]]), 0)  # Torch.combinations will make sure that there is no self-pair.

                # feature
                if relation_token is not None:
                    relation_feat = torch.cat([
                        obj_token[node_pairs[:, 0]],
                        obj_token[node_pairs[:, 1]],
                        relation_token.view(1, -1).expand(node_pairs.size(0), -1)
                    ], dim=1)
                else:
                    relation_feat = torch.cat([
                        obj_token[node_pairs[:, 0]],
                        obj_token[node_pairs[:, 1]]
                    ], dim=1)

                relation_feat = rln_proj(relation_feat)  # (n, d)
                # ================= get relation feature done =================
                relation_logits = torch.einsum("a d, b d -> a b", relation_feat, encoded_text)  # (n, d), (text_length, d) -> (n, text_length)
                relation_logits.masked_fill(~text_mask, float('-inf'))

                node_pairs = node_pairs.cpu()
                relation_prob = relation_logits.sigmoid().detach().cpu()

                all_relation = torch.zeros((relation_prob.shape[0], len(name2predicates)))
                for ii in range(1, len(sep_idx)):
                    right_idx = sep_idx[ii]
                    left_idx = sep_idx[ii-1] + 1
                    if left_idx >= right_idx:
                        continue
                    name = tokenizer.decode(input_ids[left_idx:right_idx])
                    all_relation[:, name2predicates[name]] = relation_prob[:, left_idx:right_idx].mean(-1)

                unchanged_relation = copy.deepcopy(all_relation)
                unchanged_node_pairs = copy.deepcopy(node_pairs)

                # sort by score: relation score * subject score * object score
                rel_score = all_relation[:, 1:].max(1)[0]

                obj_score0 = pred_boxes_score[node_pairs[:, 0]]
                obj_score1 = pred_boxes_score[node_pairs[:, 1]]
                rel_score = rel_score.to(obj_score0.device) * obj_score0 * obj_score1

                rel_idx = rel_score.sort(descending=True)[1].to(all_relation.device)
                all_relation = all_relation[rel_idx]
                node_pairs = node_pairs[rel_idx]

                monitor_pack = {
                    "pred_class": pred_boxes_class,
                    "pred_boxes": origin_output_boxes,
                    "rln_pred_probs": unchanged_relation,
                    "node_pairs": unchanged_node_pairs,
                    "rel_idx": rel_idx
                }
                self.monitor.record(monitor_pack, targets[0])

            else:
                assert node_id.nelement() == 1, "#obj != 1"
                print("Warning: #obj==1!")
                node_pairs = torch.zeros(1, 2).long()
                all_relation = torch.zeros(1, 51) # 
                relation_feat = None
                pred_boxes = pred_boxes.view(1, -1).repeat(2, 1)
                pred_boxes_score = pred_boxes_score.view(1, -1).repeat(2, 1)
                pred_boxes_class = pred_boxes_class.view(1, -1).repeat(2, 1)
                #pred_boxes_score.fill_(0.)

            out = {}
            if all_relation is not None:
                out['node_id'] = node_id.cpu()
                out['pred_boxes'] = pred_boxes.cpu()
                out['pred_boxes_score'] = pred_boxes_score.cpu()
                out['pred_boxes_class'] = pred_boxes_class.cpu()

                out['all_node_pairs'] = node_pairs
                out['all_relation'] = all_relation

            dst.append(out)

        return dst


class ModelPerfomanceMonitor():
    '''
    This monitor is build to monitor the performance of the OvSGTRRlnDecoderPairPruning model.
    To see how GDINO, pair selector and rln decoder performs.

    GDINO: how many potential pairs are included in top-100 objects from all objects.
    pair selector: how many potential pairs are included in top-100 pairs from top-100 objects.
    rln decoder: how many pairs are predicted correctlly in top-100 pairs.
    '''
    def __init__(self):
        self.all_pairs_inputed = 0
        self.gdino_caught_pairs = 0
        self.pair_selector_caught_pairs = 0
        self.rln_decoder_caught_pairs = 0

    def record(self, outputs, target):
        '''
        Takes outputs, which carry the GDINO, pair-selector and relation-decoder stages, and compares each against targets to report per-module recall
        monitor_pack = {
            "pred_class": pred_boxes_class,
            "pred_boxes": pred_boxes,
            "rln_pred_probs": all_relation,
            "node_pairs": node_pairs,
            "rel_idx": rel_idx,
        }

        targets:
            target:
                boxes: (n, 4)
                labels: (n)
                edges: (m, 3)
        '''
        if "__background__" not in list(self.name2classes.keys()):
            self.name2classes['__background__'] = 0

        def cxcywh_to_xyxy(boxes):
            # boxes: (..., 4)
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            return torch.stack([x1, y1, x2, y2], dim=-1)

        def box_iou_hit_matrix(boxes1, boxes2, iou_threshold=0.5):
            """
            boxes1: (n, 4)  cxcywh (normalized)
            boxes2: (m, 4)  cxcywh (normalized)
            return: (n, m) bool
            """

            # 1. convert to xyxy
            boxes1 = cxcywh_to_xyxy(boxes1)
            boxes2 = cxcywh_to_xyxy(boxes2)

            # 2. broadcast to compute the intersection
            boxes1 = boxes1[:, None, :]  # (n, 1, 4)
            boxes2 = boxes2[None, :, :]  # (1, m, 4)

            inter_x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
            inter_y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
            inter_x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
            inter_y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])

            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h

            # 3. areas
            area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
            area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

            union = area1 + area2 - inter_area

            # 4. IoU
            iou = inter_area / (union + 1e-6)

            # 5. threshold
            return iou > iou_threshold

        # pred_logits = outputs['pred_logits']
        # # pred_labels = outputs['pred_labels']
        # pred_boxes = outputs['pred_boxes']
        # entity_mask = outputs['entity_mask']
        # input_ids = outputs['input_ids']
        # batch_entity_num = (~entity_mask).sum(dim=1).tolist()

        # pair_indices = outputs['pair_indices']
        # pair_mask = outputs['pair_mask']
        # batch_pair_num = (~pair_mask).sum(dim=1).tolist()

        # rln_pred_logits = outputs['rln_pred_logits']
        # rln_input_ids = outputs['rln_input_ids']

        pred_class = outputs['pred_class']  # cuda
        pred_boxes = outputs['pred_boxes']  # cuda
        rln_pred_probs = outputs['rln_pred_probs'].to(pred_class.device)
        node_pairs = outputs['node_pairs'].to(pred_class.device)
        rel_idx = outputs['rel_idx'].to(pred_class.device)

        # GDINO hit count.
        gt_boxes = torch.tensor(target['boxes']).to(pred_class.device)
        gt_labels = torch.tensor(target['labels']).to(pred_class.device)
        gt_edges = torch.tensor(target['edges']).to(pred_class.device)

        # sep_idx = [j for j in range(len(input_ids[i])) if input_ids[i][j] in [101, 102, 1012]]
        # entity_prob = pred_logits[i].sigmoid()
        # all_entity = torch.zeros((entity_prob.shape[0], len(self.name2classes))).to(pred_logits.device)  # (num_q, category_num)
        # for ii in range(1, len(sep_idx)):
        #     right_idx = sep_idx[ii]
        #     left_idx = sep_idx[ii-1] + 1
        #     if left_idx >= right_idx:
        #         continue
        #     name = self.tokenizer.decode(input_ids[i][left_idx:right_idx])
        #     all_entity[:, self.name2classes[name]] = entity_prob[:, left_idx:right_idx].mean(-1)
        pred_labels = pred_class  # (num_q,)

        entity_box_hit_matrix = box_iou_hit_matrix(pred_boxes, gt_boxes)  # (n, m)
        entity_label_hit_matrix = pred_labels[:, None] == gt_labels[None, :]  # (n, m)
        entity_hit_matrix = entity_box_hit_matrix & entity_label_hit_matrix  # (n, m)

        entity_hit_flag = entity_hit_matrix.any(dim=0)  # (n, m) -> (m,)
        pair_hit_flag = entity_hit_flag[gt_edges[:, 0]] & entity_hit_flag[gt_edges[:, 1]]

        self.all_pairs_inputed += pair_hit_flag.shape[0]
        self.gdino_caught_pairs += pair_hit_flag.sum().item()

        # pair selector hit count.
        entity_relation_matrix = torch.full((len(pred_labels), len(pred_labels)), -1).to(pred_class.device)
        entity_relation_label_matrix = torch.full((len(pred_labels), len(pred_labels)), -1).to(pred_class.device)  # (n, n)
        # gt_edges_hit_sub_obj_index_list = []
        for ii, gt_edge in enumerate(gt_edges):
            sub_hit_index = (entity_hit_matrix[:, gt_edge[0]] != 0).nonzero(as_tuple=True)[0]
            obj_hit_index = (entity_hit_matrix[:, gt_edge[1]] != 0).nonzero(as_tuple=True)[0]

            sub_hit_index, obj_hit_index = torch.meshgrid(sub_hit_index, obj_hit_index, indexing='ij')
            sub_hit_index = sub_hit_index.reshape(-1)
            obj_hit_index = obj_hit_index.reshape(-1)

            entity_relation_matrix[sub_hit_index, obj_hit_index] = ii
            entity_relation_label_matrix[sub_hit_index, obj_hit_index] = gt_edge[2]

            # gt_edges_hit_sub_obj_index_list.append((sub_hit_index, obj_hit_index))

        rln_pred_probs = rln_pred_probs[rel_idx][:100]
        node_pairs = node_pairs[rel_idx][:100]

        max_vals, max_indices = rln_pred_probs.max(dim=1)
        pred_rln_labels = max_indices

        # max_vals, max_indices = rln_pred_probs.max(dim=1)
        # topk_vals, topk_indices = torch.topk(max_vals, 100)
        # pred_rln_labels = max_indices[topk_indices]
        # node_pairs = node_pairs[topk_indices]

        # pred_rln_labels = torch.argmax(rln_pred_probs, dim=1)  # (num_q,)
        # gt_rln_labels = entity_relation_label_matrix[node_pairs[:, 0], node_pairs[:, 1]]  # (num_q,)
        # label_hit_flag = pred_rln_labels == gt_rln_labels  # (num_q,)
        # hit_edge_indexs = entity_relation_matrix[node_pairs[:, 0], node_pairs[:, 1]]
        # hit_edge_indexs = hit_edge_indexs[label_hit_flag]
        # rln_decoder_hit_edge_num = hit_edge_indexs[hit_edge_indexs != -1].unique().numel()
        # self.rln_decoder_caught_pairs += rln_decoder_hit_edge_num

        for gt_edge in gt_edges:
            label_hit_flag = pred_rln_labels == gt_edge[2].unsqueeze(0)  # (k)
            # label_hit_flag = pred_rln_labels == gt_edge[2]  # (k)
            candidate_pairs = node_pairs[label_hit_flag]

            candidate_subject_boxes = pred_boxes[candidate_pairs[:, 0]]
            candidate_object_boxes = pred_boxes[candidate_pairs[:, 1]]
            candidate_subject_labels = pred_class[candidate_pairs[:, 0]]
            candidate_object_labels = pred_class[candidate_pairs[:, 1]]

            gt_subject_box = gt_boxes[gt_edge[0]]
            gt_object_box = gt_boxes[gt_edge[1]]
            gt_subject_label = gt_labels[gt_edge[0]]
            gt_object_label = gt_labels[gt_edge[1]]

            subject_box_hit_flag = box_iou_hit_matrix(candidate_subject_boxes, gt_subject_box.unsqueeze(0)).squeeze(1)  # (n, m)
            object_box_hit_flag = box_iou_hit_matrix(candidate_object_boxes, gt_object_box.unsqueeze(0)).squeeze(1)
            subject_label_hit_flag = candidate_subject_labels == gt_subject_label.unsqueeze(0)
            object_label_hit_flag = candidate_object_labels == gt_object_label.unsqueeze(0)

            all_hit_flag = subject_box_hit_flag & object_box_hit_flag & subject_label_hit_flag & object_label_hit_flag

            if all_hit_flag.any():
                self.rln_decoder_caught_pairs += 1

            # entity_hit_matrix (n, m)
            # subject_box_hit_matrix = box_iou_hit_matrix(candidate_subject_boxes, gt_boxes)

    def summarize(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # to tensor, so it can be all_reduced
        stats = torch.tensor([
            self.all_pairs_inputed,
            self.gdino_caught_pairs,
            self.rln_decoder_caught_pairs
        ], dtype=torch.float64, device=device)

        # DDP sync
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        # read the values back
        all_pairs_inputed = stats[0].item()
        gdino_caught_pairs = stats[1].item()
        rln_decoder_caught_pairs = stats[2].item()

        # guard against division by zero
        gdino_recall = gdino_caught_pairs / all_pairs_inputed if all_pairs_inputed > 0 else 0.0
        rln_decoder_recall = (rln_decoder_caught_pairs / gdino_caught_pairs if gdino_caught_pairs > 0 else 0.0)

        # format the output
        summary = (
            "\n========== Model Performance Summary ==========\n"
            f"Total GT Pairs           : {int(all_pairs_inputed)}\n"
            f"GDINO Caught             : {int(gdino_caught_pairs)} | Recall: {gdino_recall:.4f}\n"
            f"Relation Decoder Caught  : {int(rln_decoder_caught_pairs)} | Recall: {rln_decoder_recall:.4f}\n"
            "================================================\n"
        )

        return summary


@MODULE_BUILD_FUNCS.registe_with_name(module_name="ovsgtr_base")
def build_ovsgtr_base(args):
    backbone = build_backbone(args)
    frozen_backbone = getattr(args, "frozen_backbone", False)
    if frozen_backbone:
        print("*"*10, " Freeze backbone !", "*"*10)
        for name, p in backbone.named_parameters():
            p.requires_grad = False 

    transformer = build_transformer(args)
    monitor = ModelPerfomanceMonitor()

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    rln_freq_bias = getattr(args, "rln_freq_bias", None)
    if rln_freq_bias is not None:
        rln_freq_bias = torch.load(rln_freq_bias)

    args.text_encoder_type = '/workspace/weights/bert-base-uncased'
    model = OvSGTRBase(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        sgg_mode=args.sgg_mode, 
        num_rln_cat=args.num_rln_cat,
        rln_freq_bias=rln_freq_bias,
        num_rln_queries=args.num_rln_queries,
        focal_loss_for_edges=args.focal_loss_for_edges,
        concat_rel_text=args.concat_rel_text,
        rel_batch_per_image=args.rel_batch_per_image,
    )
    model.fix_rel_batch = False
    model.monitor = monitor

    # matcher
    matcher = build_matcher(args, tokenizer=model.tokenizer)

    model.matcher = matcher

    # loss fn
    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    weight_dict['loss_edges'] = args.edge_loss_coef
    weight_dict['loss_edges_pos'] = 1.0
    weight_dict['loss_edges_neg'] = 1.0
    weight_dict['loss_edges_fg'] = 1.0
    weight_dict['loss_distill'] = getattr(args, "distill_loss_coef", 0.1)

    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    use_masks = getattr(args, "masks", False)
    if use_masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_edges': 1.0,
            'loss_edges_pos': 1.0,
            'loss_edges_neg': 1.0,
            'loss_edges_fg': 1.0,
            'loss_obj_ll': 1.0,
            'loss_distill': 1.0, 
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes' ]
    if use_masks:
        losses += ["masks"]
    losses += ["edges"]

    criterion = OvSGTRBaseCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             tokenizer=model.tokenizer, 
                             focal_loss_for_edges=getattr(args, "focal_loss_for_edges", False),
                             rel_proposals_threshold=getattr(args, "rel_proposals_threshold", 0.5),
                             rel_proposals_threshold_enabled=getattr(args, "rel_proposals_threshold_enabled", False),
                             unsupervised_distill=getattr(args, "unsupervised_distill", False),
                             fix_rel_batch=getattr(args, "fix_rel_batch", False),
                             rel_batch_per_image=getattr(args, "rel_batch_per_image", 64),
                             )

    # post 
    use_text_labels = getattr(args, "use_text_labels", True)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, 
                        nms_iou_threshold=args.nms_iou_threshold,
                        use_text_labels=use_text_labels,
                        tokenizer = model.tokenizer,
                        detections_per_img=getattr(args, "detections_per_img", 100),
                        temperature=getattr(args, "obj_temp", 1.0) / args.hidden_dim,
                        max_text_len=getattr(args, "max_text_len", 2048),
                        use_gt_box=getattr(args, "use_gt_box", False)
                    )}
    if postprocessors['bbox'].use_gt_box:
        print("*"*10, " PostProcessing use GT Boxes !")
        postprocessors['bbox'].matcher = matcher
    postprocessors['bbox'].rln_proj = model.rln_proj
    postprocessors['bbox'].monitor = monitor

    # frozen bert
    for name, p in model.named_parameters():
        if 'bert' in name:
            p.requires_grad = False 

    if args.dn_number <= 0:
        model.label_enc.weight.requires_grad = False 
        print("disable label_enc !")


    # bb_all, bert_all, other_all = 0, 0, 0
    # bb_train, bert_train, other_train = 0, 0, 0

    # for name, p in model.named_parameters():
        
    #     # grouping
    #     if name.startswith("backbone"):
    #         bb_all += p.numel()
    #         if p.requires_grad:
    #             bb_train += p.numel()

    #     elif name.startswith("bert"):
    #         bert_all += p.numel()
    #         if p.requires_grad:
    #             bert_train += p.numel()

    #     else:
    #         other_all += p.numel()
    #         if p.requires_grad:
    #             other_train += p.numel()

    # print("=== Backbone ===")
    # print(f"all: {bb_all/1e6:.2f}M | trainable: {bb_train/1e6:.2f}M")

    # print("=== BERT ===")
    # print(f"all: {bert_all/1e6:.2f}M | trainable: {bert_train/1e6:.2f}M")

    # print("=== Other ===")
    # print(f"all: {other_all/1e6:.2f}M | trainable: {other_train/1e6:.2f}M")

    return model, criterion, postprocessors
