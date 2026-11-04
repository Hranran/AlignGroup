modelname = "ovsgtr_rln_decoder_pair_pruning_v2"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 512 #256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True


# train
frozen_weights = None
frozen_backbone = True


lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [20, 23]

aux_loss = True
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0

cls_loss_coef = 2.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

# for dn
use_dn = False
dn_number = 0
masks = False
use_text_labels = True


vg_roidb_key = 'split_GLIPunseen'
num_select = 300  # Decides how many boxes will be remained for
nms_iou_threshold = 0.5  # Whether and how many nms is used on OBD reference.
detections_per_img = 100  # Decides how many detected boxes will being remained in the end.

has_bbox_supervision = True
rln_pretraining = False
edge_loss_coef = 1.0

rln_add_coords = True


# ================ Training mission settings ================
# V6 finetune recipe: body at 1e-7, relation modules x100 => 1e-5.
lr = 1e-7
lr_rules = {
    "rln_decoder": 100.0,
    "rln_text_proj": 100.0,
    "rln_classifier": 100.0,
    "pair_selector": 100.0,
    "rln_class_embed": 100.0,
    "rln_bbox_embed": 100.0,
    "rln_interm_bbox_embed": 100.0,
    "rln_interm_class_embed": 100.0
}
weight_dict = {
    'loss_ce': cls_loss_coef,
    'loss_bbox': bbox_loss_coef,
    'loss_giou': giou_loss_coef,
    # selector not used to pick pairs in all_pairs_grid mode -> not trained.
    'loss_pair_pruning': 0.0,
    'loss_rln_bbox': 1.0,
    'loss_rln_giou': 1.0,
    'loss_rln_labels': 1.0,
    # V6-style anti-forgetting distillation (only active when teacher_weight
    # is set; the V2 build reads exactly these two keys from this dict).
    'loss_distill': 0.0,
    'loss_encoder_distill': 0.0,
}
coeff_weight_dict = {
    'loss_ce': 1.0,
    'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
    'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
    'loss_rln_bbox': 1.0,
    'loss_rln_giou': 1.0,
    'loss_rln_labels': 1.0,
    'loss_obj_ll': 1.0,
    'loss_distill': 1.0,
}

weight_decay = 0.0001
batch_size = 4   # all-pairs decoder + a second (teacher) forward per iter.
                 # V6 finetuned at 5, but V2's n^2-pair decoder is much heavier;
                 # raise back toward 4 if memory allows.
epochs = 18

sgg_mode = 'ovdr'  # 'pretrain', 'full', 'full_text', 'ovd', 'ovr', 'ovdr'.
sg_ovd_mode = True  # filter out novel entity categories at train
sg_ovr_mode = True  # filter out novel relation categories at train
focal_loss_for_edges = True  # Loss set, whether to use focal loss or not.
rel_batch_per_image = 64  # gt_entity loss batch: positives + sampled negatives.
pretrain_data_percentage = 1.0
# TODO: point to the E7 pretrain checkpoint on the target machine. Student
# init (--pretrain_model_path) should be the SAME checkpoint (V6 recipe).
use_distill = False
teacher_weight = None
num_rln_cat = 51  # Used in 'full' sgg mode.

# ================ Evaluation settings ================
use_test_set = False
unsupervised_distill = True   # distill without novel names; base-only rel captions
distill_loss_coef = 0.1
use_gt_box = False
rln_eval_use_obj_score = True
rln_eval_fg_filter = False   # no-op in open-vocab eval (label 0 can never score >0)
rln_eval_v6_topk   = True    # V6-style multi-class flat topk at eval

# ================ model arch settings ================
# OvSGTR's relation query feature part.
num_rln_queries = 0
concat_rel_text = True

# pair selector part
selector_loss_mode = 'all_pairs'
detach_pair_selector = True
pair_selector_topk = 1000
sep_rln_query = True   # REQUIRED: rebuilds rln_query to match the grid pair_indices.
'''
pair_indices_sample_mode controls what pair indices decoder will see during forward.
all_pairs_grid: full n*n grid (row-major, diagonal kept-but-masked).
                REQUIRED for rln_mask_mode='subject_grouped'.
'''
topk_pair_num = 1000
pair_indices_sample_mode = 'all_pairs_grid'

# Relation Decoder Part
rln_decoder_self_attn_gate_init = 1.0
rln_decoder_do_self_attention = True          # grouped self-attn ON (E6/E7)
rln_mask_mode = 'subject_grouped'             # memory-bounded grouped self-attn
rln_decoder_use_self_attn_gate = True
rln_decoder_hidden_dim = 256
rln_decoder_dim_ffn = 2048
rln_decoder_dropout = 0.0
rln_decoder_activation = "relu"
rln_decoder_num_features_layer = 4
rln_decoder_n_heads = 8
rln_decoder_n_points = 4
rln_decoder_use_text_cross_attention = True
rln_decoder_layers = 3
rln_decoder_num_feature_levels = 4

# ================ Relation-label loss sampling ================
rln_pos_source = 'gt_query'
rln_loss_sample_mode = 'gt_entity'
# E7 parity: duplicates between GT slots and topk fill are allowed. Flip to
# True only as a separate ablation.
rln_gt_query_dedup = False
