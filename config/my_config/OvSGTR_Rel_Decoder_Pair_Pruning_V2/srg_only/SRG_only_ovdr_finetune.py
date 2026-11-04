# SRG-only VG OvD+R; initialize BOTH student and teacher from its OWN pretrain.
_base_ = '../mimic_attn/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_attn_ovdr_finetune.py'
rln_decoder_use_text_cross_attention = False
rln_decoder_use_image_cross_attention = False
# Deliberately nonexistent: override only after SRG-only val selection.
# A truthy placeholder fails rather than silently running without distillation.
teacher_weight = 'weights/SELECTED_PRETRAIN_CHECKPOINT_REQUIRED.pth'
