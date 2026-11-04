# ============================================================================
# EXPERIMENT 2: all-pairs (grid) + subject-grouped self-attn + zero-init gate
#               (full pretrain)
# Purpose:
#   - Differs from Exp 1 ONLY by self-attn -> the clean "does attn help?" toggle,
#     in the cleanest regime (train/test consistent, no selector, no reweight).
#   - Zero-init gate => at iter 0 this model is identical to Exp 1, so any later
#     divergence is attributable to attn (not to "more training").
# Diff vs Exp 1 (everything else identical):
#   - rln_decoder_do_self_attention : False -> True
#   - rln_mask_mode : 'default' -> 'subject_grouped'
#   - rln_decoder_use_self_attn_gate : False -> True
# ============================================================================
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
# Keep identical to Exp 1 so the only difference is self-attn.
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
lr = 5e-6
lr_rules = {
    "rln_decoder": 10.0,
    "rln_text_proj": 10.0,
    "rln_classifier": 10.0,
    "pair_selector": 10.0,
    "rln_class_embed": 10.0,
    "rln_bbox_embed": 10.0,
    "rln_interm_bbox_embed": 10.0,
    "rln_interm_class_embed": 10.0
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
    'loss_distill': 0.1,
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
batch_size = 4   # all-pairs => ~detections_per_img^2 pairs through the decoder.
                 # grouped self-attn is memory-bounded; if OOM, the driver is the
                 # decoder cross-attn/FFN over ~10000 tokens -> lower batch_size.
epochs = 18

sgg_mode = 'pretrain'  # 'pretrain', 'full', 'full_text', 'ovd', 'ovr', 'ovdr'.
sg_ovd_mode = False  # Used for vg dataset, decides whether filter out novel entity category or not.
sg_ovr_mode = False  # Used for vg dataset, decides whether filter out novel relation category or not.
focal_loss_for_edges = True  # Loss set, whether to use focal loss or not.
rel_batch_per_image = 64  # The number of maximum relation selected during training.
pretrain_data_percentage = 1.0  # Decides how many percentages of train data will be used.
teacher_weight = None
num_rln_cat = 51  # Used in 'full' sgg mode.

# ================ Evaluation settings ================
use_test_set = False
unsupervised_distill = True
distill_loss_coef = 0.1
use_gt_box = False

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
rln_decoder_do_self_attention = True          # <-- Exp 2: self-attn ON
rln_mask_mode = 'subject_grouped'             # memory-bounded grouped self-attn
rln_decoder_use_self_attn_gate = True         # zero-init gate: starts == Exp 1
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
