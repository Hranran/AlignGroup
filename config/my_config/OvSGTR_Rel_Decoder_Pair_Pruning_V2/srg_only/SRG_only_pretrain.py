# SRG-only: remove decoder cross-modal updates; retain grouped SA + FFN.
# Inherit the E6 recipe rather than maintaining a drifting full config copy.
# epochs remains the original 18-epoch schedule. Actual pretraining budget
# is pending confirmation; an 8-epoch run matches the completed E6 budget.
_base_ = '../mimic_attn/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_attn_pretrain.py'
rln_decoder_use_text_cross_attention = False
rln_decoder_use_image_cross_attention = False
