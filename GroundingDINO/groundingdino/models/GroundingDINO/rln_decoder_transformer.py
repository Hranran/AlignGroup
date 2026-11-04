# from typing import Optional

# import torch
# import torch.utils.checkpoint as checkpoint
# from torch import Tensor, nn

# from groundingdino.util.misc import inverse_sigmoid

# from .fuse_modules import BiAttentionBlock
# from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
# from .transformer_vanilla import TransformerEncoderLayer
# from .utils import (
#     MLP,
#     _get_activation_fn,
#     _get_clones,
#     gen_encoder_output_proposals,
#     gen_sineembed_for_position,
#     get_sine_pos_embed,
# )


# class RelationDecoderLayer(nn.Module):
#     def __init__(
#         self,
#         d_model=256,
#         d_ffn=1024,
#         dropout=0.1,
#         n_heads=8,
#         n_levels=4,
#         n_points=4,
#     ):
#         super().__init__()

#         # Self-Attn
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         # Image Cross-Attn (our version)
#         self.cross_attn_image = MSDeformAttn(
#             embed_dim=d_model,
#             num_heads=n_heads,
#             num_levels=n_levels,
#             num_points=n_points,
#             batch_first=False,  # important
#         )
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)

#         # Text Cross-Attn
#         self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout3 = nn.Dropout(dropout)

#         # FFN
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.norm4 = nn.LayerNorm(d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.dropout5 = nn.Dropout(dropout)

#         self.activation = F.relu

#     def forward(
#         self,
#         tgt,                        # (bs, num_q, d)
#         query_pos,                  # (bs, num_q, d)
#         reference_points,           # (bs, num_q, n_levels, 2 or 4)
#         memory,                     # (bs, lvl_flat, d)
#         memory_key_padding_mask,    # (bs, lvl_flat)
#         spatial_shapes,
#         level_start_index,
#         memory_text,                # (bs, text_len, d)
#         text_attention_mask,        # (bs, text_len)
#     ):
#         bs, num_q, d = tgt.shape

#         # =========================
#         # 1. Self Attention
#         # =========================
#         q = k = (tgt + query_pos).transpose(0, 1)  # (num_q, bs, d)
#         v = tgt.transpose(0, 1)

#         tgt2 = self.self_attn(q, k, v)[0].transpose(0, 1)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         # =========================
#         # 2. Image Cross Attention(Deformable)
#         # =========================
#         q = (tgt + query_pos).transpose(0, 1)      # (num_q, bs, d)
#         v = memory.transpose(0, 1)                 # (lvl_flat, bs, d)

#         tgt2 = self.cross_attn_image(
#             query=q,
#             value=v,
#             key_padding_mask=memory_key_padding_mask,
#             reference_points=reference_points,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#         )

#         tgt2 = tgt2.transpose(0, 1)  # back to (bs, num_q, d)

#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)

#         # =========================
#         # 3. Text Cross Attention
#         # =========================
#         q = (tgt + query_pos).transpose(0, 1)        # (num_q, bs, d)
#         k = memory_text.transpose(0, 1)              # (text_len, bs, d)

#         tgt2 = self.cross_attn_text(
#             q,
#             k,
#             value=k,
#             key_padding_mask=text_attention_mask
#         )[0].transpose(0, 1)

#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)

#         # =========================
#         # 4. FFN
#         # =========================
#         tgt2 = self.linear2(self.dropout4(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout5(tgt2)
#         tgt = self.norm4(tgt)

#         return tgt


# class RelationDecoder(nn.Module):
#     def __init__(
#         self,
#         decoder_layer,
#         num_layers,
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList(
#             [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
#         )
#         self.num_layers = num_layers

#     def forward(
#         self,
#         query,                      # (bs, num_q, d)
#         query_pos,                  # (bs, num_q, d)
#         reference_points,           # (bs, num_q, n_levels, 2)
#         memory,
#         memory_key_padding_mask,
#         spatial_shapes,
#         level_start_index,
#         memory_text,
#         text_attention_mask,
#     ):
#         output = query

#         intermediate = []

#         for layer in self.layers:
#             output = layer(
#                 tgt=output,
#                 query_pos=query_pos,
#                 reference_points=reference_points,
#                 memory=memory,
#                 memory_key_padding_mask=memory_key_padding_mask,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 memory_text=memory_text,
#                 text_attention_mask=text_attention_mask,
#             )
#             intermediate.append(output)

#         return torch.stack(intermediate)  # (num_layers, bs, num_q, d)


# class RelationDecoderNew(nn.Module):
#     def __init__(
#         self,
#         decoder_layer,
#         num_layers,
#         norm=None,
#         return_intermediate=False,
#         d_model=256,
#         query_dim=4,
#         num_feature_levels=1,
#     ):
#         super().__init__()
#         if num_layers > 0:
#             self.layers = _get_clones(decoder_layer, num_layers)
#         else:
#             self.layers = []
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate
#         assert return_intermediate, "support return_intermediate only"
#         self.query_dim = query_dim
#         assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
#         self.num_feature_levels = num_feature_levels

#         self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
#         self.query_pos_sine_scale = None

#         self.query_scale = None

#         _bbox_embed = MLP(d_model, d_model, 4, 3)
#         box_embed_layerlist = nn.ModuleList([_bbox_embed for i in range(num_layers)])
#         self.bbox_embed = box_embed_layerlist
#         self.class_embed = None

#         self.d_model = d_model

#     def forward(
#         self,
#         tgt,
#         memory,
#         tgt_mask: Optional[Tensor] = None,
#         memory_mask: Optional[Tensor] = None,
#         tgt_key_padding_mask: Optional[Tensor] = None,
#         memory_key_padding_mask: Optional[Tensor] = None,
#         pos: Optional[Tensor] = None,
#         refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
#         # for memory
#         level_start_index: Optional[Tensor] = None,  # num_levels
#         spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
#         valid_ratios: Optional[Tensor] = None,
#         # for text
#         memory_text: Optional[Tensor] = None,
#         text_attention_mask: Optional[Tensor] = None,
#     ):
#         """
#         Input:
#             - tgt: nq, bs, d_model
#             - memory: hw, bs, d_model
#             - pos: hw, bs, d_model
#             - refpoints_unsigmoid: nq, bs, 2/4.
#             - valid_ratios/spatial_shapes: bs, nlevel, 2
#         """
#         output = tgt

#         intermediate = []
#         reference_points = refpoints_unsigmoid.sigmoid()
#         ref_points = [reference_points]

#         for layer_id, layer in enumerate(self.layers):
#             reference_points_input = (reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :])  # nq, bs, nlevel, 4
#             query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2

#             # conditional query
#             query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256

#             # main process
#             output = layer(
#                 tgt=output,
#                 tgt_query_pos=query_pos,
#                 tgt_query_sine_embed=query_sine_embed,
#                 tgt_key_padding_mask=tgt_key_padding_mask,
#                 tgt_reference_points=reference_points_input,
#                 memory_text=memory_text,
#                 text_attention_mask=text_attention_mask,
#                 memory=memory,
#                 memory_key_padding_mask=memory_key_padding_mask,
#                 memory_level_start_index=level_start_index,
#                 memory_spatial_shapes=spatial_shapes,
#                 memory_pos=pos,
#                 self_attn_mask=tgt_mask,
#                 cross_attn_mask=memory_mask,
#             )
#             if output.isnan().any() | output.isinf().any():
#                 print(f"output layer_id {layer_id} is nan")
#                 try:
#                     num_nan = output.isnan().sum().item()
#                     num_inf = output.isinf().sum().item()
#                     print(f"num_nan {num_nan}, num_inf {num_inf}")
#                 except Exception as e:
#                     print(e)

#             # iter update
#             reference_before_sigmoid = inverse_sigmoid(reference_points)
#             delta_unsig = self.bbox_embed[layer_id](output)
#             outputs_unsig = delta_unsig + reference_before_sigmoid
#             new_reference_points = outputs_unsig.sigmoid()

#             reference_points = new_reference_points.detach()
#             # if layer_id != self.num_layers - 1:
#             ref_points.append(new_reference_points)

#             intermediate.append(self.norm(output))

#         return [
#             [itm_out.transpose(0, 1) for itm_out in intermediate],
#             [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
#         ]


# class DeformableTransformerDecoderLayer(nn.Module):
#     def __init__(
#         self,
#         d_model=256,
#         d_ffn=1024,
#         dropout=0.1,
#         activation="relu",
#         n_levels=4,
#         n_heads=8,
#         n_points=4,
#         use_text_feat_guide=False,
#         use_text_cross_attention=False,
#         key_padding_mask_for_self_attn=False,
#         use_lvl_pos=False,
#         do_self_attention=True
#     ):
#         super().__init__()

#         # cross attention
#         self.cross_attn = MSDeformAttn(
#             embed_dim=d_model,
#             num_levels=n_levels,
#             num_heads=n_heads,
#             num_points=n_points,
#             batch_first=True,
#         )
#         self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.norm1 = nn.LayerNorm(d_model)

#         # cross attention text
#         if use_text_cross_attention:
#             self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#             self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#             self.catext_norm = nn.LayerNorm(d_model)

#         # self attention
#         if do_self_attention:
#             self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#             self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#             self.norm2 = nn.LayerNorm(d_model)
#         else:
#             self.self_attn = None

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
#         self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.norm3 = nn.LayerNorm(d_model)

#         self.key_aware_proj = None
#         self.use_text_feat_guide = use_text_feat_guide
#         assert not use_text_feat_guide
#         self.use_text_cross_attention = use_text_cross_attention

#         self.key_padding_mask_for_self_attn = key_padding_mask_for_self_attn
#         self.use_lvl_pos = use_lvl_pos

#     def rm_self_attn_modules(self):
#         self.self_attn = None
#         self.dropout2 = None
#         self.norm2 = None

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, tgt):
#         with torch.cuda.amp.autocast(enabled=False):
#             tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt

#     def forward(
#         self,
#         # for tgt
#         tgt: Optional[Tensor],  # nq, bs, d_model
#         tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
#         tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
#         tgt_key_padding_mask: Optional[Tensor] = None,
#         tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
#         memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
#         text_attention_mask: Optional[Tensor] = None,  # bs, num_token
#         # for memory
#         memory: Optional[Tensor] = None,  # hw, bs, d_model
#         memory_key_padding_mask: Optional[Tensor] = None,
#         memory_level_start_index: Optional[Tensor] = None,  # num_levels
#         memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
#         memory_pos: Optional[Tensor] = None,  # pos for memory
#         # sa
#         self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
#         cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
#     ):
#         """
#         Input:
#             - tgt/tgt_query_pos: nq, bs, d_model
#             -
#         """
#         assert cross_attn_mask is None

#         # self attention
#         if self.self_attn is not None:
#             # import ipdb; ipdb.set_trace()
#             q = k = self.with_pos_embed(tgt, tgt_query_pos)
#             if not self.key_padding_mask_for_self_attn:
#                 tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
#             else:
#                 tgt2 = self.self_attn(q, k, tgt, key_padding_mask=self_attn_mask)[0]
#             assert not torch.isnan(tgt2).any()
#             tgt = tgt + self.dropout2(tgt2)
#             tgt = self.norm2(tgt)

#         if self.use_text_cross_attention:
#             tgt2 = self.ca_text(
#                 self.with_pos_embed(tgt, tgt_query_pos),
#                 memory_text.transpose(0, 1),
#                 memory_text.transpose(0, 1),
#                 key_padding_mask=text_attention_mask,
#             )[0]
#             tgt = tgt + self.catext_dropout(tgt2)
#             tgt = self.catext_norm(tgt)

#         tgt2 = self.cross_attn(
#             query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
#             reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
#             value=memory.transpose(0, 1) if not self.use_lvl_pos else self.with_pos_embed(memory, memory_pos).transpose(0, 1),
#             spatial_shapes=memory_spatial_shapes,
#             level_start_index=memory_level_start_index,
#             key_padding_mask=memory_key_padding_mask,
#         ).transpose(0, 1)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         # ffn
#         tgt = self.forward_ffn(tgt)

#         return tgt


# def pair_box_to_reference_points(pair_box, num_levels):
#     """
#     pair_box: (bs, num_q, 2, 4)  # cxcywh (normalized)
#     return:   (bs, num_q, num_levels, 4)
#     """
#     sub = pair_box[:, :, 0]  # (bs, num_q, 4)
#     obj = pair_box[:, :, 1]

#     # cxcywh → xyxy
#     def cxcywh_to_xyxy(box):
#         cx, cy, w, h = box.unbind(-1)
#         x1 = cx - 0.5 * w
#         y1 = cy - 0.5 * h
#         x2 = cx + 0.5 * w
#         y2 = cy + 0.5 * h
#         return torch.stack([x1, y1, x2, y2], dim=-1)

#     sub_xyxy = cxcywh_to_xyxy(sub)
#     obj_xyxy = cxcywh_to_xyxy(obj)

#     # union
#     x1 = torch.min(sub_xyxy[..., 0], obj_xyxy[..., 0])
#     y1 = torch.min(sub_xyxy[..., 1], obj_xyxy[..., 1])
#     x2 = torch.max(sub_xyxy[..., 2], obj_xyxy[..., 2])
#     y2 = torch.max(sub_xyxy[..., 3], obj_xyxy[..., 3])

#     # back to cxcywh
#     cx = (x1 + x2) / 2
#     cy = (y1 + y2) / 2
#     w = (x2 - x1)
#     h = (y2 - y1)

#     union_box = torch.stack([cx, cy, w, h], dim=-1)  # (bs, num_q, 4)

#     # expand to levels
#     reference_points = union_box.unsqueeze(2).repeat(1, 1, num_levels, 1)

#     return reference_points


# def build_relation_decoder(args):
#     rln_decoder_layer = DeformableTransformerDecoderLayer(
#         d_model=args.rln_decoder_hidden_dim,
#         d_ffn=args.rln_decoder_dim_ffn,
#         dropout=args.rln_decoder_dropout,
#         activation=args.rln_decoder_activation,
#         n_levels=args.rln_decoder_num_features_layer,
#         n_heads=args.rln_decoder_n_heads,
#         n_points=args.rln_decoder_n_points,
#         use_text_cross_attention=args.rln_decoder_use_text_cross_attention,
#         key_padding_mask_for_self_attn=True,
#     )

#     rln_decoder = RelationDecoderNew(
#         decoder_layer=rln_decoder_layer,
#         num_layers=args.rln_decoder_layers,
#         norm=nn.LayerNorm(args.rln_decoder_hidden_dim),
#         return_intermediate=True,
#         d_model=args.rln_decoder_hidden_dim,
#         query_dim=4,
#         num_feature_levels=args.rln_decoder_num_feature_levels,
#     )
#     return rln_decoder


from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn

from groundingdino.util.misc import inverse_sigmoid

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .transformer_vanilla import TransformerEncoderLayer
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)


class RelationDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        n_levels=4,
        n_points=4,
    ):
        super().__init__()

        # Self-Attn
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Image Cross-Attn (our version)
        self.cross_attn_image = MSDeformAttn(
            embed_dim=d_model,
            num_heads=n_heads,
            num_levels=n_levels,
            num_points=n_points,
            batch_first=False,  # important
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Text Cross-Attn
        self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        tgt,                        # (bs, num_q, d)
        query_pos,                  # (bs, num_q, d)
        reference_points,           # (bs, num_q, n_levels, 2 or 4)
        memory,                     # (bs, lvl_flat, d)
        memory_key_padding_mask,    # (bs, lvl_flat)
        spatial_shapes,
        level_start_index,
        memory_text,                # (bs, text_len, d)
        text_attention_mask,        # (bs, text_len)
    ):
        bs, num_q, d = tgt.shape

        # =========================
        # 1. Self Attention
        # =========================
        q = k = (tgt + query_pos).transpose(0, 1)  # (num_q, bs, d)
        v = tgt.transpose(0, 1)

        tgt2 = self.self_attn(q, k, v)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # =========================
        # 2. Image Cross Attention(Deformable)
        # =========================
        q = (tgt + query_pos).transpose(0, 1)      # (num_q, bs, d)
        v = memory.transpose(0, 1)                 # (lvl_flat, bs, d)

        tgt2 = self.cross_attn_image(
            query=q,
            value=v,
            key_padding_mask=memory_key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        tgt2 = tgt2.transpose(0, 1)  # back to (bs, num_q, d)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # =========================
        # 3. Text Cross Attention
        # =========================
        q = (tgt + query_pos).transpose(0, 1)        # (num_q, bs, d)
        k = memory_text.transpose(0, 1)              # (text_len, bs, d)

        tgt2 = self.cross_attn_text(
            q,
            k,
            value=k,
            key_padding_mask=text_attention_mask
        )[0].transpose(0, 1)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # =========================
        # 4. FFN
        # =========================
        tgt2 = self.linear2(self.dropout4(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout5(tgt2)
        tgt = self.norm4(tgt)

        return tgt


class RelationDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        query,                      # (bs, num_q, d)
        query_pos,                  # (bs, num_q, d)
        reference_points,           # (bs, num_q, n_levels, 2)
        memory,
        memory_key_padding_mask,
        spatial_shapes,
        level_start_index,
        memory_text,
        text_attention_mask,
    ):
        output = query

        intermediate = []

        for layer in self.layers:
            output = layer(
                tgt=output,
                query_pos=query_pos,
                reference_points=reference_points,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
            )
            intermediate.append(output)

        return torch.stack(intermediate)  # (num_layers, bs, num_q, d)


class RelationDecoderNew(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None

        _bbox_embed = MLP(d_model, d_model, 4, 3)
        box_embed_layerlist = nn.ModuleList([_bbox_embed for i in range(num_layers)])
        self.bbox_embed = box_embed_layerlist
        self.class_embed = None

        self.d_model = d_model

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        rln_self_attn_mask = None,
        subject_group_n: Optional[int] = None,
        subject_group_mask: Optional[Tensor] = None,
        group_axis: str = 'subject',
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4.
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            reference_points_input = (reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :])  # nq, bs, nlevel, 4
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2

            # conditional query
            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                key_padding_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                rln_self_attn_mask=rln_self_attn_mask,
                subject_group_n=subject_group_n,
                subject_group_mask=subject_group_mask,
                group_axis=group_axis,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)

            # iter update
            reference_before_sigmoid = inverse_sigmoid(reference_points)
            delta_unsig = self.bbox_embed[layer_id](output)
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = outputs_unsig.sigmoid()

            reference_points = new_reference_points.detach()
            # if layer_id != self.num_layers - 1:
            ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
        key_padding_mask_for_self_attn=False,
        use_lvl_pos=False,
        do_self_attention=True,
        use_self_attn_gate=False,
        self_attn_gate_init=0.0,
        use_image_cross_attention=True,
    ):
        super().__init__()

        # Default True preserves Full/E5 construction, RNG order and state keys.
        # SRG-only removes this entire update, including its residual norm.
        self.use_image_cross_attention = use_image_cross_attention
        if use_image_cross_attention:
            self.cross_attn = MSDeformAttn(
                embed_dim=d_model,
                num_levels=n_levels,
                num_heads=n_heads,
                num_points=n_points,
                batch_first=True,
            )
            self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.norm1 = nn.LayerNorm(d_model)
        else:
            self.cross_attn = None
            self.dropout1 = None
            self.norm1 = None

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        if do_self_attention:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.self_attn = None

        # optional zero-init gate on the self-attn residual.
        # When enabled, the layer starts strictly equal to the no-self-attn variant
        # (gate=0), so self-attn can only be learned in, never destroy the init.
        # Default False -> no extra parameter is created -> state_dict unchanged.
        self.use_self_attn_gate = use_self_attn_gate
        if do_self_attention and use_self_attn_gate:
            # init 0.0 -> attn starts dormant (clean attribution, but slow to wake);
            # init 1.0 -> attn active from step 0 (lets attn weights actually train).
            self.self_attn_gate = nn.Parameter(torch.full((1,), float(self_attn_gate_init)))

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

        self.key_padding_mask_for_self_attn = key_padding_mask_for_self_attn
        self.use_lvl_pos = use_lvl_pos

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _subject_grouped_self_attn(self, q, k, v, n, group_mask, group_axis='subject'):
        """
        Memory-bounded self-attention restricted to pairs sharing an entity.

        The full pair set is the (n x n) relation grid in row-major order, i.e.
        the flat query index p == sub * n + obj. Instead of building an
        (n*n) x (n*n) attention matrix (which OOMs), we fold the grouping entity
        into the batch axis and run a length-n attention per group. Attention
        scores cost O(bs * n * n^2) instead of O(bs * (n^2)^2).

        group_axis selects which entity defines a group:
          'subject' (default): pairs sharing a subject attend (over the object
                     axis) -- the row of the grid; this is our proposed model.
          'object' : pairs sharing an object attend (over the subject axis) --
                     the column of the grid; used only for the ablation that
                     tests whether subject grouping is the right choice.

        Args:
            q, k, v:    (n*n, bs, d)  seq-first tokens, flat index = sub*n + obj.
            n:          int, number of entities (grid side length).
            group_mask: (bs, n, n) bool, True = invalid (sub,obj) key
                        (diagonal / padded entity), indexed [b, sub, obj].

        Returns:
            tgt2: (n*n, bs, d)  self-attention output in the same layout as q.
        """
        nq, bs, d = q.shape
        assert nq == n * n, f"grouped self-attn expects n*n tokens, got nq={nq}, n={n}"

        if group_axis == 'subject':
            # group over objects per (batch, subject); batch index = b*n + sub, seq = obj
            def to_groups(x):
                return x.reshape(n, n, bs, d).permute(1, 2, 0, 3).reshape(n, bs * n, d)
            def from_groups(out):
                return out.reshape(n, bs, n, d).permute(2, 0, 1, 3).reshape(n * n, bs, d)
            kpm = group_mask.reshape(bs * n, n).clone()          # [b*n+sub, obj]
        elif group_axis == 'object':
            # group over subjects per (batch, object); batch index = b*n + obj, seq = sub
            def to_groups(x):
                return x.reshape(n, n, bs, d).permute(0, 2, 1, 3).reshape(n, bs * n, d)
            def from_groups(out):
                return out.reshape(n, bs, n, d).permute(0, 2, 1, 3).reshape(n * n, bs, d)
            kpm = group_mask.permute(0, 2, 1).reshape(bs * n, n).clone()  # [b*n+obj, sub]
        else:
            raise ValueError(f"group_axis must be 'subject' or 'object', got {group_axis}")

        qg, kg, vg = to_groups(q), to_groups(k), to_groups(v)

        # a fully-masked group (padded entity / single-entity image) would make
        # softmax produce NaN; let it attend freely, its output is dropped
        # downstream by pair_mask.
        fully_masked = kpm.all(dim=1)
        kpm[fully_masked] = False

        out = self.self_attn(qg, kg, vg, key_padding_mask=kpm)[0]  # (n, bs*n, d)
        return from_groups(out)

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        key_padding_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
        rln_self_attn_mask = None,
        subject_group_n: Optional[int] = None,      # set -> use grouped self-attn
        subject_group_mask: Optional[Tensor] = None,  # (bs, n, n) bool, True = invalid key
        group_axis: str = 'subject',                 # 'subject' (ours) or 'object' (ablation)
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            # if not self.key_padding_mask_for_self_attn:
            #     tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            # else:
            #     tgt2 = self.self_attn(q, k, tgt, key_padding_mask=self_attn_mask)[0]
            if subject_group_n is not None:
                # memory-bounded grouped self-attn (all-pairs grid)
                tgt2 = self._subject_grouped_self_attn(q, k, tgt, subject_group_n, subject_group_mask, group_axis)
            else:
                if rln_self_attn_mask is not None:
                    rln_self_attn_mask = rln_self_attn_mask.repeat_interleave(self.self_attn.num_heads, dim=0)
                tgt2 = self.self_attn(q, k, tgt, key_padding_mask=key_padding_mask, attn_mask=rln_self_attn_mask)[0]
            assert not torch.isnan(tgt2).any()
            if self.use_self_attn_gate:
                tgt = tgt + self.self_attn_gate * self.dropout2(tgt2)
            else:
                tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        if self.use_image_cross_attention:
            tgt2 = self.cross_attn(
                query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
                value=memory.transpose(0, 1) if not self.use_lvl_pos else self.with_pos_embed(memory, memory_pos).transpose(0, 1),
                spatial_shapes=memory_spatial_shapes,
                level_start_index=memory_level_start_index,
                key_padding_mask=memory_key_padding_mask,
            ).transpose(0, 1)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def pair_box_to_reference_points(pair_box, num_levels):
    """
    pair_box: (bs, num_q, 2, 4)  # cxcywh (normalized)
    return:   (bs, num_q, num_levels, 4)
    """
    sub = pair_box[:, :, 0]  # (bs, num_q, 4)
    obj = pair_box[:, :, 1]

    # cxcywh → xyxy
    def cxcywh_to_xyxy(box):
        cx, cy, w, h = box.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    sub_xyxy = cxcywh_to_xyxy(sub)
    obj_xyxy = cxcywh_to_xyxy(obj)

    # union
    x1 = torch.min(sub_xyxy[..., 0], obj_xyxy[..., 0])
    y1 = torch.min(sub_xyxy[..., 1], obj_xyxy[..., 1])
    x2 = torch.max(sub_xyxy[..., 2], obj_xyxy[..., 2])
    y2 = torch.max(sub_xyxy[..., 3], obj_xyxy[..., 3])

    # back to cxcywh
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1)
    h = (y2 - y1)

    union_box = torch.stack([cx, cy, w, h], dim=-1)  # (bs, num_q, 4)

    # expand to levels
    reference_points = union_box.unsqueeze(2).repeat(1, 1, num_levels, 1)

    return reference_points


def build_relation_decoder(args):
    rln_decoder_layer = DeformableTransformerDecoderLayer(
        d_model=args.rln_decoder_hidden_dim,
        d_ffn=args.rln_decoder_dim_ffn,
        dropout=args.rln_decoder_dropout,
        activation=args.rln_decoder_activation,
        n_levels=args.rln_decoder_num_features_layer,
        n_heads=args.rln_decoder_n_heads,
        n_points=args.rln_decoder_n_points,
        use_text_cross_attention=args.rln_decoder_use_text_cross_attention,
        use_image_cross_attention=getattr(args, "rln_decoder_use_image_cross_attention", True),
        key_padding_mask_for_self_attn=True,
        do_self_attention=getattr(args, "rln_decoder_do_self_attention", True),
        use_self_attn_gate=getattr(args, "rln_decoder_use_self_attn_gate", False),
        self_attn_gate_init=getattr(args, "rln_decoder_self_attn_gate_init", 0.0),
    )

    rln_decoder = RelationDecoderNew(
        decoder_layer=rln_decoder_layer,
        num_layers=args.rln_decoder_layers,
        norm=nn.LayerNorm(args.rln_decoder_hidden_dim),
        return_intermediate=True,
        d_model=args.rln_decoder_hidden_dim,
        query_dim=4,
        num_feature_levels=args.rln_decoder_num_feature_levels,
    )
    return rln_decoder

