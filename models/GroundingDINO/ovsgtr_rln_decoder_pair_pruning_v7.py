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
This variant exists to find out why adding pair pruning in V2 broke the relation decoder.
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
from .transformer_my import build_transformer
from .rln_decoder_transformer import build_relation_decoder
from .utils import MLP, ContrastiveEmbed, gen_sineembed_for_position

from .utils import FrequencyBias

# 
import bisect
from .matcher import build_matcher 
from .losses import OvSGTRRelDecoderPairPruningCriterionV7
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


class OvSGTRRlnDecoderPairPruning(nn.Module):
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
        rln_decoder=None,
        use_box_score=False,
        pair_selector_topk=100,
        sep_rln_query=False,
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

        # global rln queries init
        self.num_rln_queries = num_rln_queries
        if self.num_rln_queries > 0:
            self.rln_embed = nn.Embedding(self.num_rln_queries, self.hidden_dim*2)
            self.transformer.decoder.rln_embed = self.rln_embed
        else:
            self.transformer.decoder.rln_embed = None

        # rln text porj init
        self.rln_text_proj = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)

        # rln pair selector init
        # self.pair_selector = PairSelector(hidden_dim=hidden_dim, k=pair_selector_topk, use_box_score=use_box_score)

        self.sep_rln_query = sep_rln_query
        if self.sep_rln_query:
            self.rln_proj = MLP(self.hidden_dim * 2, 2048, self.hidden_dim, 2) # 2048->1024

        # rln_decoder init
        input_dim = self.hidden_dim * (2 + self.num_rln_queries)
        # self.rln_proj = MLP(input_dim, 2048, self.hidden_dim, 2) # 2048->1024
        self.rln_decoder = rln_decoder

        # rln_bbox_embed init
        _rln_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_rln_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_rln_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(rln_decoder.num_layers)]
        self.rln_bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.rln_decoder.bbox_embed = self.rln_bbox_embed

        self.rln_interm_bbox_embed = copy.deepcopy(_rln_bbox_embed)

        # rln_class_embed init
        _rln_class_embed = ContrastiveEmbed(max_text_len=self.max_text_len) # 2048
        rln_class_embed_layerlist = [_rln_class_embed for i in range(rln_decoder.num_layers)]
        self.rln_class_embed = nn.ModuleList(rln_class_embed_layerlist)
        self.rln_decoder.class_embed = self.rln_class_embed

        self.rln_interm_class_embed = copy.deepcopy(_rln_class_embed)

        # full classifier init
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

        self._positive_map = None

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

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
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

        # hs, hs_rln, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
        #     srcs, masks, input_query_bbox, poss, 
        #     input_query_label, attn_mask, text_dict,
        # )
        transformer_output_dict = self.transformer(
            srcs, masks, input_query_bbox, poss, 
            input_query_label, attn_mask, text_dict,
        )
        hs = transformer_output_dict['hs']
        hs_rln = transformer_output_dict['hs_rln']
        reference = transformer_output_dict['references']
        hs_enc = transformer_output_dict['hs_enc']
        ref_enc = transformer_output_dict['ref_enc']
        init_box_proposal = transformer_output_dict['init_box_proposal']

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
        gdino_output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1], "object_token": hs_obj[-1]}

        # For Hun match.
        out['text_dict'] = text_dict
        input_ids = text_dict['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]

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
        edge_gt_lbls, edge_gt_boxes, rln_out = self.sgg_forward(
            gdino_output=gdino_output,
            memory=transformer_output_dict['memory'],
            memory_key_padding_mask=transformer_output_dict['memory_key_padding_mask'],
            memory_pos=transformer_output_dict['memory_pos'],
            spatial_shapes=transformer_output_dict['spatial_shapes'],
            level_start_index=transformer_output_dict['level_start_index'],
            valid_ratios=transformer_output_dict['valid_ratios'],
            rln_text_dict=rel_text_dict,
            targets=targets
            )

        out['rel_text_dict'] = rel_text_dict
        out['edge_gt_lbls'] = edge_gt_lbls
        out['edge_gt_boxes'] = edge_gt_boxes
        out['rln_out'] = rln_out

        # =============== monitor info preparation ===============
        if not self.training:
            rln_out['monitor_pack']['input_ids'] = text_dict['input_ids']
            self.monitor.record(rln_out['monitor_pack'], targets)
        # GDINO: top-100 entity's
        #     pred_labels: predicted labels, still to be mapped  (b, k)
        #     pred_boxes: (b, k, 4)
        #     entity_mask: (b, k)
        # pair_selector: top-100 pairs, how many GDINO got but not pair_selector
        #     pair_indices: (b, k, 2)
        #     pair_mask: (b, k)
        # rln_decoder:
        #     rln_pred_logits: (b, k, text_length)
        #     input_ids: (b, text_length)

        # if not self.training:
        #     out['postprocess_pack'] = {
        #         'obj_token': hs_obj[-1],
        #         'memory': transformer_output_dict['memory'],
        #         'memory_key_padding_mask': transformer_output_dict['memory_key_padding_mask'],
        #         'memory_pos': transformer_output_dict['memory_pos'],
        #         'spatial_shapes': transformer_output_dict['spatial_shapes'],
        #         'level_start_index': transformer_output_dict['level_start_index'],
        #         'valid_ratios': transformer_output_dict['valid_ratios']
        #     }

        return out

    def sgg_forward(self, gdino_output, memory, memory_key_padding_mask, memory_pos, spatial_shapes, level_start_index, valid_ratios, rln_text_dict, targets):
        """
        sample pairs.
        """
        # ===================== box quality filter =====================
        # gdino_output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1], "object_token": hs_obj[-1]}
        pred_logits = gdino_output['pred_logits'].detach()
        pred_boxes = gdino_output['pred_boxes'].detach()
        entity_tokens = gdino_output['object_token']

        # caption_list = [target['caption'] for target in targets]
        def get_positive_map(cat_list=None, max_text_len=512):
            if cat_list is None:
                if self._positive_map is not None:
                    return self._positive_map
                cat_list = [e for e in self.name2classes.keys()]
                use_name2class = True
            else:
                use_name2class = False

            captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
            tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
            positive_map = create_positive_map_from_span(self.tokenizer(captions), tokenspanlist, max_text_len)  # (num_cat, max_text_length)          

            if use_name2class:
                max_label = max(self.name2classes.values()) + 1  # name2classes, [__background__] not included.
                new_pos_map = torch.zeros((max_label, positive_map.shape[1]))  # (num_cat + 1, max_text_length)
                for ii, (name, label) in enumerate(self.name2classes.items()):
                    new_pos_map[label] = positive_map[ii]  # (num_cat + 1, max_text_length)
                self._positive_map = new_pos_map
                return self._positive_map

            return positive_map  # (num_cat, max_text_length)

        def split_to_cat_list(caption):
            cat_list = caption.split(". ")
            cat_list = [x.strip().lower() for x in cat_list]
            # assert len(cat_list) == 100, print(caption)
            return cat_list
        positive_map = torch.stack([get_positive_map(cat_list=split_to_cat_list(target['caption'])).to(pred_logits.device) for target in targets])  # (b, num_cat, text_length)

        pred_probs = pred_logits.sigmoid()
        pred_probs = torch.einsum('b q t, b c t -> b q c', pred_probs, positive_map)  # (b, q, c)

        scores, labels = pred_probs.max(dim=-1)  # (b, num_query), (b, num_query)
        topk_scores, topk_indices = scores.topk(300, dim=-1)  # (b, k), (b, k)
        topk_labels = torch.gather(labels, 1, topk_indices)  # (b, k), these labels are NOT neccesarrily equal across batch, since every batch's caption is different.
        topk_boxes = torch.gather(pred_boxes, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, 4))  # (b, k, 4)
        topk_entity_tokens = torch.gather(entity_tokens, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, entity_tokens.shape[-1]))
        topk_logits = torch.gather(pred_logits, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, pred_logits.shape[-1]))

        item_indices = [batched_nms(b, s, l, iou_threshold=0.5) for b,s,l in zip(box_ops.box_cxcywh_to_xyxy(topk_boxes), topk_scores, topk_labels)]
        item_indices = [e[:100] for e in item_indices]  # (100)

        filtered_pred_scores = [s[i] for s, i in zip(topk_scores, item_indices)]
        filtered_pred_logits = [l[i] for l, i in zip(topk_logits, item_indices)]
        filtered_pred_boxes = [b[i] for b, i in zip(topk_boxes, item_indices)]
        filtered_entity_tokens = [t[i] for t, i in zip(topk_entity_tokens, item_indices)]

        def build_entity_mask(entity_nums):
            """
            vectorized version
            """
            entity_nums = torch.tensor(entity_nums)
            max_q = entity_nums.max()

            mask = torch.arange(max_q)[None, :] >= entity_nums[:, None]
            return mask
        batch_entity_num = [len(l) for l in filtered_pred_logits]
        entity_mask = build_entity_mask(batch_entity_num).to(filtered_pred_logits[0].device)

        filtered_pred_scores = torch.nn.utils.rnn.pad_sequence(filtered_pred_scores, batch_first=True)  # (b, n)
        filtered_pred_logits = torch.nn.utils.rnn.pad_sequence(filtered_pred_logits, batch_first=True)  # (b, n, text_length)
        filtered_pred_boxes = torch.nn.utils.rnn.pad_sequence(filtered_pred_boxes, batch_first=True)  # (b, n, 4)
        filtered_entity_tokens = torch.nn.utils.rnn.pad_sequence(filtered_entity_tokens, batch_first=True)  # (b, n, d)

        b, n, d = filtered_pred_logits.shape

        # ===================== Hungarian matching sampler =====================
        # Map gt entities into filtered entities.
        indices = self.matcher({"pred_logits": filtered_pred_logits, "pred_boxes": filtered_pred_boxes}, targets)  # b of list, [b][0]: query index  [b][1]: gt box index
    
        entity_gt_boxes = torch.zeros((b, n, 4)).to(gdino_output['object_token'].device)
        # rln_label_matrix = torch.zeros((b, n, n)).to(gdino_output['object_token'].device)
        rln_label_matrix = torch.zeros((b, n, n), dtype=torch.long, device=gdino_output['object_token'].device)
        for i, target in enumerate(targets):
            tgt_edges = target['edges']
            tgt_boxes = target['boxes']  # (n, 4)
            gt_to_query_mapping = {}
            for query, gt in zip(indices[i][0].tolist(), indices[i][1].tolist()):
                gt_to_query_mapping[gt] = query
                entity_gt_boxes[i, query] = tgt_boxes[gt]

            for tgt_edge in tgt_edges:
                rln_label_matrix[i, gt_to_query_mapping[tgt_edge[0].item()], gt_to_query_mapping[tgt_edge[1].item()]] = tgt_edge[2]

        # pair selector
        # rln_query, pair_indices, rln_exist_score, pair_mask = self.pair_selector(filtered_entity_tokens, entity_mask, filtered_pred_scores)  # (b, k, d), (b, k, 2), (b, k), (b, k)

        row = torch.arange(n).unsqueeze(1).expand(n, n).to(filtered_pred_logits[0].device)
        col = torch.arange(n).unsqueeze(0).expand(n, n).to(filtered_pred_logits[0].device)

        pair_indices = torch.stack([row.reshape(-1), col.reshape(-1)], dim=-1).to(filtered_pred_logits[0].device)  # (n * n, 2)
        pair_indices = pair_indices.unsqueeze(0).expand(b, -1, -1)  # (b, n*n, 2)

        # generate mask
        # (1, n*n)
        row_flat = row.reshape(-1).unsqueeze(0)
        col_flat = col.reshape(-1).unsqueeze(0)

        # (b, 1)
        valid_num = torch.tensor(batch_entity_num).to(filtered_pred_logits[0].device).unsqueeze(1)

        # padding mask
        invalid_row = row_flat >= valid_num
        invalid_col = col_flat >= valid_num

        # diagonal mask
        diag_mask = row_flat == col_flat

        # final mask
        pair_mask = diag_mask | invalid_row | invalid_col


        sub_idx = pair_indices[..., 0]  # (b, k)
        obj_idx = pair_indices[..., 1]  # (b, k)

        # gather subject features
        sub_feat = torch.gather(filtered_entity_tokens, dim=1, index=sub_idx.unsqueeze(-1).expand(-1, -1, filtered_entity_tokens.shape[2]))  # (b, k, d)
        # gather object features
        obj_feat = torch.gather(filtered_entity_tokens, dim=1, index=obj_idx.unsqueeze(-1).expand(-1, -1, filtered_entity_tokens.shape[2]))  # (b, k, d)
        # concat
        rln_query = torch.cat([sub_feat, obj_feat], dim=-1)  # (b, k, 2d)
        rln_query = self.rln_proj(rln_query)

        # sample gt
        batch_idx = torch.arange(b).unsqueeze(1).expand(-1, rln_query.shape[1])  # (b, k)
        rln_gt_labels = rln_label_matrix[batch_idx, pair_indices[..., 0], pair_indices[..., 1]]  # (b, k)
        sub_gt_boxes = torch.gather(entity_gt_boxes, dim=1, index=sub_idx.unsqueeze(-1).expand(-1, -1, 4))  # (b, k, 4)
        obj_gt_boxes = torch.gather(entity_gt_boxes, dim=1, index=obj_idx.unsqueeze(-1).expand(-1, -1, 4))  # (b, k, 4) 

        # get feature for rln decoder
        subject_boxes = torch.gather(filtered_pred_boxes, dim=1, index=sub_idx.unsqueeze(-1).expand(-1, -1, 4))  # (b, k, 4)
        object_boxes = torch.gather(filtered_pred_boxes, dim=1, index=obj_idx.unsqueeze(-1).expand(-1, -1, 4))  # (b, k, 4)

        # ======================= rln_decoder part =======================
        def box_cxcywh_to_xyxy(box):
            cx, cy, w, h = box.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            return torch.stack([x1, y1, x2, y2], dim=-1)

        def box_xyxy_to_cxcywh(box):
            x1, y1, x2, y2 = box.unbind(-1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)
            return torch.stack([cx, cy, w, h], dim=-1)

        def get_union_box(sub_box, obj_box):
            sub_xyxy = box_cxcywh_to_xyxy(sub_box)
            obj_xyxy = box_cxcywh_to_xyxy(obj_box)

            x1 = torch.min(sub_xyxy[..., 0], obj_xyxy[..., 0])
            y1 = torch.min(sub_xyxy[..., 1], obj_xyxy[..., 1])
            x2 = torch.max(sub_xyxy[..., 2], obj_xyxy[..., 2])
            y2 = torch.max(sub_xyxy[..., 3], obj_xyxy[..., 3])

            union_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            union_cxcywh = box_xyxy_to_cxcywh(union_xyxy)
            return union_cxcywh

        # rln_query_mask = pair_mask

        union_boxes = get_union_box(subject_boxes, object_boxes)
        union_boxes = union_boxes.clamp(min=0.01, max=1 - 0.01)
        union_boxes_unsigmoid = torch.log(union_boxes / (1 - union_boxes)) # unsigmoid
        # union_boxes_unsigmoid = union_boxes_unsigmoid.masked_fill(rln_query_mask.unsqueeze(-1), float('inf'))
        assert not torch.isnan(union_boxes_unsigmoid).any()

        rln_hs, rln_references = self.rln_decoder(
            tgt=rln_query.transpose(0, 1),  # (batch, num_q, hidden_dim)  (4, 900, 256)
            memory=memory.transpose(0, 1),  # (bs, lvl_flat, hidden_dim)  (4, 17029, 256)
            memory_key_padding_mask=memory_key_padding_mask,  # (bs, lvl_flat)  (4, 17029)
            pos=memory_pos.transpose(0, 1),  # (bs, lvl_flat, hidden_dim) (4, 17029, 256)
            refpoints_unsigmoid=union_boxes_unsigmoid.transpose(0, 1),  # (bs, num_q, 4) (4, 900, 4)
            level_start_index=level_start_index,  # (level_num) (4)
            spatial_shapes=spatial_shapes,  # (level_num, 2) (4, 2) shape of each level
            valid_ratios=valid_ratios,  # (bs, level_num, 2) (4, 4, 2) valid (non-masked) wh ratio of each level
            tgt_mask=pair_mask,  # (bs, num_q)
            memory_text=rln_text_dict["encoded_text"],  # (bs, text_length, hidden_dim) (4, 304, 256)
            text_attention_mask=~rln_text_dict["text_token_mask"],  # (bs, text_length) (4, 304)
            # we ~ the mask . False means use the token; True means pad the token
        )

        # ==================== prepare output ====================
        # decoder layer bbox output
        rln_outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(rln_references[:-1], self.rln_bbox_embed, rln_hs)):  # rln_references: the ref output of every layer, including the layer-0 input. rln_hs: the tgt output of every layer, excluding the layer-0 input. The box is this layer's delta plus this layer's input ref, hence the offset
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            rln_outputs_coord_list.append(layer_outputs_unsig)
        rln_outputs_coord_list = torch.stack(rln_outputs_coord_list)

        # decoder layer class logits output
        rln_outputs_class = torch.stack([layer_cls_embed(layer_hs, rln_text_dict) for layer_cls_embed, layer_hs in zip(self.rln_class_embed, rln_hs)])
        rln_out = {
            "rln_pred_logits": rln_outputs_class[-1],  # (b, n, text_length)
            "rln_pred_boxes": rln_outputs_coord_list[-1],
            # "rln_query_mask": rln_query_mask,
            "batch_pair_num": (~pair_mask).sum(dim=1).tolist(),
            # "rln_exist_score": rln_exist_score
            }
        # aux outputs
        rln_out['aux_outputs'] = [{"rln_pred_logits": a, "rln_pred_boxes": b} for a, b in zip(rln_outputs_class[:-1], rln_outputs_coord_list[:-1])]

        # interm outputs
        interm_coord = (self.rln_interm_bbox_embed(rln_query) + union_boxes_unsigmoid).sigmoid().unsqueeze(0)
        interm_class = self.rln_interm_class_embed(rln_query, rln_text_dict)
        rln_out['interm_outputs'] = {'rln_pred_logits': interm_class, 'rln_pred_boxes': interm_coord}

        edge_gt_boxes = get_union_box(sub_gt_boxes, obj_gt_boxes)

        # THIS INFERENCE IS NOT COMPATIBLE WITH COCO PRETRAIN
        if not self.training:  # If not self.training, include sgg infer result into rln_out.
            # ==================== sgg_evaluator used output ====================
            positive_map = get_positive_map().to(filtered_pred_logits.device)  # (num_class + 1, max_text_length), under reference, every caption should be the cat of every cat and be the same.
            # topk_logits: (b, k, max_text_length)
            pred_probs = torch.einsum('b k t, c t -> b k c', filtered_pred_logits.sigmoid(), positive_map)  # (b, k, c)
            scores, labels = pred_probs.max(dim=-1)
            boxes = box_ops.box_cxcywh_to_xyxy(filtered_pred_boxes)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            img_h, img_w = orig_target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            results = []
            for i, i_batch_entity_num in enumerate(batch_entity_num):
                results.append(
                    {
                        'scores': scores[i, :i_batch_entity_num],
                        'labels': labels[i, :i_batch_entity_num],
                        'boxes': boxes[i, :i_batch_entity_num, :],
                    }
                    )

            input_ids = rln_text_dict['input_ids'][0]  # Under infer settings, all batch's caption should be the same.

            # (b, k, c)
            relation_probs = torch.zeros((rln_outputs_class[-1].shape[0], rln_outputs_class[-1].shape[1], len(self.name2predicates)))
            relation_probs_text = rln_outputs_class[-1].sigmoid()
            sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]
            for ii in range(1, len(sep_idx)):
                right_idx = sep_idx[ii]
                left_idx = sep_idx[ii-1] + 1
                if left_idx >= right_idx:
                    continue
                name = self.tokenizer.decode(input_ids[left_idx:right_idx])
                relation_probs[:, :, self.name2predicates[name]] = relation_probs_text[:, :, left_idx:right_idx].mean(-1)

            batch_pair_num = (~pair_mask).sum(dim=1).tolist()
            # Each batch needs a sgg_results stores in 'graph'
            sgg_results = []
            for i, (i_batch_entity_num, i_batch_pair_num) in enumerate(zip(batch_entity_num, batch_pair_num)):
                # Do a ranking on all_node_pairs and all_relation
                all_relation = relation_probs[i, :i_batch_pair_num, :]
                rel_score = all_relation[:, 1:].max(1)[0]

                pred_boxes_score = scores[i, :i_batch_pair_num]
                node_pairs = pair_indices[i, :i_batch_pair_num, :]
                obj_score0 = pred_boxes_score[node_pairs[:, 0]]
                obj_score1 = pred_boxes_score[node_pairs[:, 1]]
                # rel_score = rel_score * obj_score0 * obj_score1  # with obj score or not?

                rel_idx = rel_score.sort(descending=True)[1].to(all_relation.device)
                all_relation = all_relation[rel_idx]
                node_pairs = node_pairs[rel_idx]

                sgg_results.append(
                    {
                        'node_id': torch.arange(i_batch_entity_num),
                        'pred_boxes': boxes[i, :i_batch_entity_num, :],
                        'pred_boxes_score': pred_boxes_score,
                        'pred_boxes_class': labels[i, :i_batch_entity_num],

                        # 'all_node_pairs': pair_indices[i, :i_batch_pair_num, :],
                        # 'all_relation': relation_probs[i, :i_batch_pair_num, :]
                        'all_node_pairs': node_pairs,
                        'all_relation': all_relation
                    }
                )

            _ = [result.update({'scale_fct': scale_fct[batch_id]}) for batch_id, result in enumerate(results)]
            for batch, res in enumerate(results):
                res.update({'graph': sgg_results[batch]})

            rln_out['sgg_infer_output'] = results

            # ==================== monitor used output ====================
            # GDINO: top-100 entity's
            #     pred_labels: predicted labels, still to be mapped  (b, k)
            #     pred_boxes: (b, k, 4)
            #     entity_mask: (b, k)
            # pair_selector: top-100 pairs, how many GDINO got but not pair_selector
            #     pair_indices: (b, k, 2)
            #     pair_mask: (b, k)
            # rln_decoder:
            #     rln_pred_logits: (b, k, text_length)
            #     input_ids: (b, text_length)

            # targets:
            #     target:
            #         boxes: (n, 4)
            #         labels: (n)
            #         edges: (m, 3)
            monitor_pack = {
                "pred_logits": filtered_pred_logits,
                "pred_boxes": filtered_pred_boxes,
                "entity_mask": entity_mask,
                "pair_indices": pair_indices,
                "pair_mask": pair_mask,
                "rln_pred_logits": rln_outputs_class[-1],
                "rln_input_ids": rln_text_dict['input_ids'],
            }
            rln_out['monitor_pack'] = monitor_pack

        return rln_gt_labels, edge_gt_boxes, rln_out


class PairSelector(nn.Module):
    def __init__(self, hidden_dim=256, k=100, use_box_score=False):
        super().__init__()
        self.k = k
        self.use_box_score = use_box_score

        self.rln_porj = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, hidden_dim),
        )

        self.rln_score_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, entity_tokens, entity_mask, pred_scores):
        """
        entity_tokens: (b, n, d)
        entity_mask:   (b, n)  True = padding
        pred_scores:   (b, n)
        """

        b, n, d = entity_tokens.shape
        device = entity_tokens.device

        # -------------------------------------------------
        # 1. build pairs (b, n, n, 2d) -> (b, n*n, 2d)
        # -------------------------------------------------
        sub_tokens = entity_tokens.unsqueeze(2).expand(b, n, n, d)
        obj_tokens = entity_tokens.unsqueeze(1).expand(b, n, n, d)
        sub_scores = pred_scores.unsqueeze(2).expand(b, n, n)
        obj_scores = pred_scores.unsqueeze(1).expand(b, n, n)

        pair_tokens = torch.cat([sub_tokens, obj_tokens], dim=-1)  # (b, n, n, 2d)
        pair_box_scores = sub_scores * obj_scores

        # record the indices
        idx = torch.arange(n, device=device)
        sub_idx = idx.view(1, n, 1).expand(b, n, n)
        obj_idx = idx.view(1, 1, n).expand(b, n, n)
        pair_indices = torch.stack([sub_idx, obj_idx], dim=-1)  # (b, n, n, 2)

        # flatten
        pair_tokens = pair_tokens.view(b, n * n, 2 * d)
        pair_indices = pair_indices.view(b, n * n, 2)
        pair_box_scores = pair_box_scores.view(b, n * n)

        # -------------------------------------------------
        # 2. build the mask (drop the diagonal + entity_mask)
        # -------------------------------------------------
        # entity mask broadcast
        sub_mask = entity_mask.unsqueeze(2).expand(b, n, n)
        obj_mask = entity_mask.unsqueeze(1).expand(b, n, n)

        # diagonal mask
        diag_mask = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0).expand(b, n, n)

        # merge the masks
        pair_mask = sub_mask | obj_mask | diag_mask  # (b, n, n)
        pair_mask = pair_mask.view(b, n * n)  # (b, n*n)

        # -------------------------------------------------
        # 3. rln token + score
        # -------------------------------------------------
        rln_tokens = self.rln_porj(pair_tokens)  # (b, n*n, d)

        scores = self.rln_score_proj(rln_tokens).squeeze(-1)  # (b, n*n)
        if self.use_box_score:
            scores = scores * pair_box_scores

        # set invalid positions to -inf so they are never selected
        scores = scores.masked_fill(pair_mask, float('-inf'))

        # -------------------------------------------------
        # 4. topk, handling padding per batch item
        # -------------------------------------------------
        out_tokens = []
        out_indices = []
        out_scores = []
        out_masks = []

        for i in range(b):
            valid_mask = ~pair_mask[i]  # (n*n)
            valid_num = valid_mask.sum()

            if valid_num > 0:
                valid_scores = scores[i][valid_mask]
                valid_tokens = rln_tokens[i][valid_mask]
                valid_indices = pair_indices[i][valid_mask]

                k_i = min(self.k, valid_num)

                topk_scores_i, topk_idx = torch.topk(valid_scores, k_i)

                topk_tokens_i = valid_tokens[topk_idx]
                topk_indices_i = valid_indices[topk_idx]

            else:
                k_i = 0
                topk_scores_i = torch.empty(0, device=device)
                topk_tokens_i = torch.empty(0, d, device=device)
                topk_indices_i = torch.empty(0, 2, dtype=torch.long, device=device)

            # -------- pad to k --------
            pad_len = self.k - k_i

            if pad_len > 0:
                pad_tokens = torch.zeros(pad_len, d, device=device)
                pad_indices = torch.zeros(pad_len, 2, dtype=torch.long, device=device)
                pad_scores = torch.full((pad_len,), float('-inf'), device=device)

                topk_tokens_i = torch.cat([topk_tokens_i, pad_tokens], dim=0)
                topk_indices_i = torch.cat([topk_indices_i, pad_indices], dim=0)
                topk_scores_i = torch.cat([topk_scores_i, pad_scores], dim=0)

                mask_i = torch.cat([
                    torch.zeros(k_i, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device)
                ], dim=0)
            else:
                mask_i = torch.zeros(self.k, dtype=torch.bool, device=device)

            out_tokens.append(topk_tokens_i)
            out_indices.append(topk_indices_i)
            out_scores.append(topk_scores_i)
            out_masks.append(mask_i)

        # stack
        pair_tokens = torch.stack(out_tokens, dim=0)     # (b, k, d)
        pair_indices = torch.stack(out_indices, dim=0)   # (b, k, 2)
        topk_scores = torch.stack(out_scores, dim=0)     # (b, k)
        pair_mask = torch.stack(out_masks, dim=0)        # (b, k)

        return pair_tokens, pair_indices, topk_scores, pair_mask


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
    def forward(self, outputs, target_sizes, cat_list=None, gt_dicts=None):
        return outputs['rln_out']['sgg_infer_output']

    def graph_infer(self, results : List[Dict], postprocess_pack, rel_text_dict):
        dst = []
        for batch_id, result in enumerate(results):
            obj_token = result['obj_token'] # (#obj, dim)

            encoded_text = rel_text_dict['encoded_text'][batch_id]
            text_mask = rel_text_dict['text_token_mask'][batch_id]
            input_ids = rel_text_dict['input_ids'][batch_id]
            sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]

            boxes = copy.deepcopy(result['boxes'])  # (#obj, 4)
            graph_infer_used_boxes = copy.deepcopy(result['graph_infer_used_bbox'])  # un-normalized cxcywh
            scores = copy.deepcopy(result['scores'])  # (#obj)
            labels = copy.deepcopy(result['labels'])  # (#obj)

            node_id = torch.nonzero(labels).squeeze() 

            obj_token = obj_token[node_id]
            pred_classes = labels[node_id]
            pred_cls_score = scores[node_id]

            pred_boxes = boxes[node_id]
            graph_infer_used_boxes = graph_infer_used_boxes[node_id]
            pred_boxes_score = pred_cls_score
            pred_boxes_class = pred_classes


            def box_cxcywh_to_xyxy(box):
                cx, cy, w, h = box.unbind(-1)
                x1 = cx - 0.5 * w
                y1 = cy - 0.5 * h
                x2 = cx + 0.5 * w
                y2 = cy + 0.5 * h
                return torch.stack([x1, y1, x2, y2], dim=-1)

            def box_xyxy_to_cxcywh(box):
                x1, y1, x2, y2 = box.unbind(-1)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = (x2 - x1)
                h = (y2 - y1)
                return torch.stack([cx, cy, w, h], dim=-1)

            def get_union_box(sub_box, obj_box):
                sub_xyxy = box_cxcywh_to_xyxy(sub_box)
                obj_xyxy = box_cxcywh_to_xyxy(obj_box)

                x1 = torch.min(sub_xyxy[..., 0], obj_xyxy[..., 0])
                y1 = torch.min(sub_xyxy[..., 1], obj_xyxy[..., 1])
                x2 = torch.max(sub_xyxy[..., 2], obj_xyxy[..., 2])
                y2 = torch.max(sub_xyxy[..., 3], obj_xyxy[..., 3])

                union_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                union_cxcywh = box_xyxy_to_cxcywh(union_xyxy)
                return union_cxcywh

            if node_id.dim() !=0 and node_id.nelement() != 0 and node_id.shape[0]>1:
                # all possible node pairs in all token ordering
                tmp = torch.arange(len(node_id))
                node_pairs = torch.cat((torch.combinations(tmp), torch.combinations(tmp)[:, [1, 0]]), 0)  # Torch.combinations will make sure that there is no self-pair.

                # feature
                relation_feat = torch.cat([obj_token[node_pairs[:, 0]], obj_token[node_pairs[:, 1]]], dim=1).unsqueeze(0)
                relation_feat = self.rln_proj(relation_feat)  # (n, d)

                sub_boxes = graph_infer_used_boxes[node_pairs[:, 0]].unsqueeze(0)
                obj_boxes = graph_infer_used_boxes[node_pairs[:, 1]].unsqueeze(0)
                union_boxes = get_union_box(sub_boxes, obj_boxes)
                union_boxes = union_boxes.clamp(min=0.01, max=1 - 0.01)
                union_boxes_unsigmoid = torch.log(union_boxes / (1 - union_boxes)) # unsigmoid

                # union_boxes = get_union_box(subject_boxes, object_boxes)
                # union_boxes = union_boxes.clamp(min=0.01, max=1 - 0.01)
                # union_boxes_unsigmoid = torch.log(union_boxes / (1 - union_boxes)) # unsigmoid
                # union_boxes_unsigmoid = union_boxes_unsigmoid.masked_fill(rln_query_mask.unsqueeze(-1), float('inf'))
                # union_boxes_unsigmoid_detached = union_boxes_unsigmoid.detach()

                relation_feat, _ = self.rln_decoder(
                    tgt=relation_feat.transpose(0, 1),  # (batch, num_q, hidden_dim)  (4, 900, 256)
                    memory=postprocess_pack['memory'].transpose(0, 1),  # (bs, lvl_flat, hidden_dim)  (4, 17029, 256)
                    memory_key_padding_mask=postprocess_pack['memory_key_padding_mask'],  # (bs, lvl_flat)  (4, 17029)
                    pos=postprocess_pack['memory_pos'].transpose(0, 1),  # (bs, lvl_flat, hidden_dim) (4, 17029, 256)
                    refpoints_unsigmoid=union_boxes_unsigmoid.transpose(0, 1),  # (bs, num_q, 4) (4, 900, 4)
                    level_start_index=postprocess_pack['level_start_index'],  # (level_num) (4)
                    spatial_shapes=postprocess_pack['spatial_shapes'],  # (level_num, 2) (4, 2) shape of each level
                    valid_ratios=postprocess_pack['valid_ratios'],  # (bs, level_num, 2) (4, 4, 2) valid (non-masked) wh ratio of each level
                    tgt_mask=None,  # (bs, num_q)
                    memory_text=encoded_text.unsqueeze(0),  # (bs, text_length, hidden_dim) (4, 304, 256)
                    text_attention_mask=~text_mask.unsqueeze(0),  # (bs, text_length) (4, 304)
                    # we ~ the mask . False means use the token; True means pad the token
                )

                # ================= get relation feature done =================
                relation_feat = relation_feat[-1].squeeze(0)

                relation_logits = torch.einsum("a d, b d -> a b", relation_feat, encoded_text)  # (n, d), (text_length, d) -> (n, text_length)
                relation_logits.masked_fill(~text_mask, float('-inf'))

                node_pairs = node_pairs.cpu()
                relation_prob = relation_logits.sigmoid().detach().cpu()

                all_relation = torch.zeros((relation_prob.shape[0], len(self.name2predicates)))
                for ii in range(1, len(sep_idx)):
                    right_idx = sep_idx[ii]
                    left_idx = sep_idx[ii-1] + 1
                    if left_idx >= right_idx:
                        continue
                    name = self.tokenizer.decode(input_ids[left_idx:right_idx])
                    all_relation[:, self.name2predicates[name]] = relation_prob[:, left_idx:right_idx].mean(-1)

                # sort by score: relation score * subject score * object score
                rel_score = all_relation[:, 1:].max(1)[0]

                obj_score0 = pred_boxes_score[node_pairs[:, 0]]
                obj_score1 = pred_boxes_score[node_pairs[:, 1]]
                rel_score = rel_score.to(obj_score0.device) * obj_score0 * obj_score1

                rel_idx = rel_score.sort(descending=True)[1].to(all_relation.device)
                all_relation = all_relation[rel_idx]
                node_pairs = node_pairs[rel_idx]
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

        self.multi_assign_count = 0
        self.total_edge_count = 0

        self.pair_selector_caught_pairs = 0
        self.rln_decoder_caught_pairs = 0

    def record(self, outputs, targets):
        '''
        Takes outputs, which carry the GDINO, pair-selector and relation-decoder stages, and compares each against targets to report per-module recall
        monitor_pack = {
            "pred_logits": filtered_pred_logits,
            "pred_boxes": filtered_pred_boxes,
            "entity_mask": entity_mask,
            "input_ids": text_dict['input_ids'],
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "rln_pred_logits": rln_outputs_class[-1],
            "rln_input_ids": rln_text_dict['input_ids'],
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

        pred_logits = outputs['pred_logits']
        # pred_labels = outputs['pred_labels']
        pred_boxes = outputs['pred_boxes']
        entity_mask = outputs['entity_mask']
        input_ids = outputs['input_ids']
        batch_entity_num = (~entity_mask).sum(dim=1).tolist()

        pair_indices = outputs['pair_indices']
        pair_mask = outputs['pair_mask']
        batch_pair_num = (~pair_mask).sum(dim=1).tolist()

        rln_pred_logits = outputs['rln_pred_logits']
        rln_input_ids = outputs['rln_input_ids']

        for i, target in enumerate(targets):
            # GDINO hit count.
            gt_boxes = torch.tensor(target['boxes']).to(pred_logits.device)
            gt_labels = torch.tensor(target['labels']).to(pred_logits.device)
            gt_edges = torch.tensor(target['edges']).to(pred_logits.device)

            sep_idx = [j for j in range(len(input_ids[i])) if input_ids[i][j] in [101, 102, 1012]]
            entity_prob = pred_logits[i].sigmoid()
            all_entity = torch.zeros((entity_prob.shape[0], len(self.name2classes))).to(pred_logits.device)  # (num_q, category_num)
            for ii in range(1, len(sep_idx)):
                right_idx = sep_idx[ii]
                left_idx = sep_idx[ii-1] + 1
                if left_idx >= right_idx:
                    continue
                name = self.tokenizer.decode(input_ids[i][left_idx:right_idx])
                all_entity[:, self.name2classes[name]] = entity_prob[:, left_idx:right_idx].mean(-1)
            pred_labels = torch.argmax(all_entity, dim=1)  # (num_q,)

            entity_box_hit_matrix = box_iou_hit_matrix(pred_boxes[i, :batch_entity_num[i], :], gt_boxes)  # (n, m)
            entity_label_hit_matrix = (pred_labels[:batch_entity_num[i]])[:, None] == gt_labels[None, :]  # (n, m)
            entity_hit_matrix = entity_box_hit_matrix & entity_label_hit_matrix  # (n, m)

            entity_hit_flag = entity_hit_matrix.any(dim=0)  # (n, m) -> (m,)
            pair_hit_flag = entity_hit_flag[gt_edges[:, 0]] & entity_hit_flag[gt_edges[:, 1]]

            self.all_pairs_inputed += pair_hit_flag.shape[0]
            self.gdino_caught_pairs += pair_hit_flag.sum().item()

            # pair selector hit count.
            entity_relation_matrix = torch.full((batch_entity_num[i], batch_entity_num[i]), -1).to(pred_logits.device)
            entity_relation_label_matrix = torch.full((batch_entity_num[i], batch_entity_num[i]), -1).to(pred_logits.device)  # (n, n)

            debug_tensor = torch.zeros((len(gt_edges)), dtype=torch.bool).to(pred_logits.device)
            debug_count = 0

            for ii, gt_edge in enumerate(gt_edges):
                sub_hit_index = (entity_hit_matrix[:, gt_edge[0]] != 0).nonzero(as_tuple=True)[0]
                obj_hit_index = (entity_hit_matrix[:, gt_edge[1]] != 0).nonzero(as_tuple=True)[0]

                sub_hit_index, obj_hit_index = torch.meshgrid(sub_hit_index, obj_hit_index, indexing='ij')
                sub_hit_index = sub_hit_index.reshape(-1)
                obj_hit_index = obj_hit_index.reshape(-1)

                debug_count += sub_hit_index.shape[0]

                if sub_hit_index.shape[0] > 0 and obj_hit_index.shape[0] > 0:
                    debug_tensor[ii] = True

                entity_relation_matrix[sub_hit_index, obj_hit_index] = ii
                entity_relation_label_matrix[sub_hit_index, obj_hit_index] = gt_edge[2]

                # gt_edges_hit_sub_obj_index_list.append((sub_hit_index, obj_hit_index))

            debug_count_2 = (entity_relation_label_matrix != -1).sum().item()

            assert torch.equal(debug_tensor, pair_hit_flag)
            # assert (debug_count == debug_count_2)

            self.total_edge_count += len(gt_edges)
            # self.multi_assign_count += (entity_relation_label_matrix != -1).sum().item()
            self.multi_assign_count += debug_count

            # i_pair_indices = pair_indices[i][:batch_pair_num[i]]
            i_pair_indices = pair_indices[i][~pair_mask[i]]
            hit_edge_indexs = entity_relation_matrix[i_pair_indices[:, 0], i_pair_indices[:, 1]]
            pair_selector_hit_edge_num = hit_edge_indexs[hit_edge_indexs != -1].unique().numel()
            self.pair_selector_caught_pairs += pair_selector_hit_edge_num

            # rln decoder hit count.
            sep_idx = [j for j in range(len(rln_input_ids[i])) if rln_input_ids[i][j] in [101, 102, 1012]]

            relation_prob = rln_pred_logits[i].sigmoid()
            all_relation = torch.zeros((relation_prob.shape[0], len(self.name2predicates))).to(pred_logits.device)  # (num_q, category_num)
            for ii in range(1, len(sep_idx)):
                right_idx = sep_idx[ii]
                left_idx = sep_idx[ii-1] + 1
                if left_idx >= right_idx:
                    continue
                name = self.tokenizer.decode(rln_input_ids[i][left_idx:right_idx])
                all_relation[:, self.name2predicates[name]] = relation_prob[:, left_idx:right_idx].mean(-1)

            pred_rln_labels = torch.argmax(all_relation, dim=1)[~pair_mask[i]]  # (num_q,)
            gt_rln_labels = entity_relation_label_matrix[i_pair_indices[:, 0], i_pair_indices[:, 1]]  # (num_q,)
            label_hit_flag = pred_rln_labels == gt_rln_labels  # (num_q,)
            hit_edge_indexs = entity_relation_matrix[i_pair_indices[:, 0], i_pair_indices[:, 1]]
            hit_edge_indexs = hit_edge_indexs[label_hit_flag]
            rln_decoder_hit_edge_num = hit_edge_indexs[hit_edge_indexs != -1].unique().numel()
            self.rln_decoder_caught_pairs += rln_decoder_hit_edge_num

    def summarize(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # to tensor, so it can be all_reduced
        stats = torch.tensor([
            self.all_pairs_inputed,
            self.gdino_caught_pairs,
            self.pair_selector_caught_pairs,
            self.rln_decoder_caught_pairs,
            self.multi_assign_count,
            self.total_edge_count
        ], dtype=torch.float64, device=device)

        # DDP sync
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        # read the values back
        all_pairs_inputed = stats[0].item()
        gdino_caught_pairs = stats[1].item()
        pair_selector_caught_pairs = stats[2].item()
        rln_decoder_caught_pairs = stats[3].item()
        multi_assign_count = stats[4].item()
        total_edge_count = stats[5].item()

        # guard against division by zero
        gdino_recall = gdino_caught_pairs / all_pairs_inputed if all_pairs_inputed > 0 else 0.0
        pair_selector_recall = (
            pair_selector_caught_pairs / gdino_caught_pairs if gdino_caught_pairs > 0 else 0.0
        )
        rln_decoder_recall = (
            rln_decoder_caught_pairs / pair_selector_caught_pairs if pair_selector_caught_pairs > 0 else 0.0
        )

        # format the output
        summary = (
            "\n========== Model Performance Summary ==========\n"
            f"Total GT Pairs           : {int(all_pairs_inputed)}\n"
            f"GDINO Caught             : {int(gdino_caught_pairs)} | Recall: {gdino_recall:.4f}\n"
            f"Pair Selector Caught     : {int(pair_selector_caught_pairs)} | Recall: {pair_selector_recall:.4f}\n"
            f"Relation Decoder Caught  : {int(rln_decoder_caught_pairs)} | Recall: {rln_decoder_recall:.4f}\n"
            f"Multi assign count       : {int(multi_assign_count)}\n"
            f"total edge count         : {int(total_edge_count)}\n"
            "================================================\n"
        )

        return summary


@MODULE_BUILD_FUNCS.registe_with_name(module_name="ovsgtr_rln_decoder_pair_pruning_v7")
def build_ovsgtr_rln_decoder_pair_pruning_v7(args):
    backbone = build_backbone(args)
    frozen_backbone = getattr(args, "frozen_backbone", False)
    if frozen_backbone:
        print("*"*10, " Freeze backbone !", "*"*10)
        for name, p in backbone.named_parameters():
            p.requires_grad = False 

    transformer = build_transformer(args)
    rln_decoder = build_relation_decoder(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    rln_freq_bias = getattr(args, "rln_freq_bias", None)
    if rln_freq_bias is not None:
        rln_freq_bias = torch.load(rln_freq_bias)

    args.text_encoder_type = '/workspace/weights/bert-base-uncased'
    model = OvSGTRRlnDecoderPairPruning(
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
        rln_decoder=rln_decoder,
        use_box_score=getattr(args, 'use_box_score', False),
        pair_selector_topk=getattr(args, 'pair_selector_topk', 100),
        sep_rln_query=getattr(args, 'sep_rln_query', False)
    )
    model.fix_rel_batch = False

    monitor = ModelPerfomanceMonitor()
    monitor.tokenizer = model.tokenizer
    model.monitor = monitor

    # matcher
    matcher = build_matcher(args, tokenizer=model.tokenizer)

    model.matcher = matcher
    # model.pair_selector.matcher = matcher

    # ================= prepare loss =================
    # base loss dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rln_bbox'] = 1.0
    weight_dict['loss_rln_giou'] = 1.0
    weight_dict['loss_rln_labels'] = 1.0

    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)


    # aux part
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # interm part
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
            'loss_rln_bbox': 1.0,
            'loss_rln_giou': 1.0,
            'loss_rln_labels': 1.0,
            'loss_obj_ll': 1.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    weight_dict.update({"loss_pair_pruning": 1.0})

    losses = ['labels', 'boxes']

    criterion = OvSGTRRelDecoderPairPruningCriterionV7(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             tokenizer=model.tokenizer, 
                             focal_loss_for_edges=getattr(args, "focal_loss_for_edges", False),
                             rel_proposals_threshold=getattr(args, "rel_proposals_threshold", 0.5),
                             rel_proposals_threshold_enabled=getattr(args, "rel_proposals_threshold_enabled", False),
                             unsupervised_distill=getattr(args, "unsupervised_distill", False),
                             fix_rel_batch=getattr(args, "fix_rel_batch", False),
                             rel_batch_per_image=getattr(args, "rel_batch_per_image", 64),
                             )
    criterion.sgg_mode = args.sgg_mode

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
    # postprocessors['bbox'].rln_proj = model.rln_proj
    postprocessors['bbox'].rln_decoder = model.rln_decoder

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
