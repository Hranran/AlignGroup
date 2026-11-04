# AlignGroup

Research code for **AlignGroup: Progressive Cross-Modal Alignment and
Subject-Relation Grouping for Open-Vocabulary SGG**.

Tengfei Gong, Zeran Huang, Yaxiong Chen, and Shengwu Xiong.

AlignGroup improves relation feature construction using Cross-Modal Relation
Aligner (CMRA) and Subject-anchored Relation Grouping (SRG). It follows the
OvSGTR data pipeline, without adding training data, VLM supervision, or
external knowledge graphs beyond that pipeline.

## Method and configurations

| Model | Text and image cross-attention | Subject-grouped self-attention | Config directory |
| --- | --- | --- | --- |
| AlignGroup | On | On | `mimic_attn/` |
| AlignGroup-CMRA | On | Off | `mimic/` |
| SRG-only | Off | On | `srg_only/` |

The directories above are under
`config/my_config/OvSGTR_Rel_Decoder_Pair_Pruning_V2/`.

- Main model: `GroundingDINO/groundingdino/models/GroundingDINO/ovsgtr_rln_decoder_pair_pruning_v2.py`.
- CMRA/SRG: `GroundingDINO/groundingdino/models/GroundingDINO/rln_decoder_transformer.py`.
- Entry point: `main_new.py`; training/evaluation loops: `engine.py`.
- `models/GroundingDINO/` is a physical copy required by the original model
  registry. Keep its model files synchronized with the canonical directory;
  `python tools/check_release.py` checks the two main files.
- Earlier model variants are retained for import compatibility. The paper's
  AlignGroup model uses `ovsgtr_rln_decoder_pair_pruning_v2`.

## Environment

The historical paper environment used Linux, Python 3.9, PyTorch 1.12.0,
torchvision 0.13.0, CUDA 11.6 and four RTX 3090 GPUs. Use a separate environment:

```bash
git clone https://github.com/Hranran/AlignGroup.git
cd AlignGroup
conda create -n aligngroup python=3.9 -y
conda activate aligngroup
bash install.sh
export PYTHONPATH="$PWD:$PWD/GroundingDINO${PYTHONPATH:+:$PYTHONPATH}"
python tools/check_release.py
python -c "import torch; import groundingdino._C; print(torch.__version__)"
```

Building Grounding DINO requires a compatible CUDA toolkit/compiler.
`requirements.txt` preserves the main historical pins and lists the runtime
imports; it is not a complete lockfile. Installing it does not by itself
verify numerical reproduction. Use this README rather than the upstream
Grounding DINO README's older PyTorch installation example.

## Initial weights

Download the [Grounding DINO Swin-T checkpoint](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
to `GroundingDINO/weights/groundingdino_swint_ogc.pth`.
Use `bert-base-uncased` from Hugging Face, or set a local directory:

```bash
export ALIGNGROUP_BERT_PATH=/path/to/bert-base-uncased
export ALIGNGROUP_ZEROSHOT_PATH=/path/to/VG/zeroshot_triplet.pytorch
```

These variables replace machine-specific paths. The VG zero-shot triplet
file is also read by the inherited GQA evaluator; use the same file/protocol
as the reference setup. The reported open-vocabulary metrics are distinct
from this inherited conventional zero-shot-triplet metric.

**Trained AlignGroup checkpoints, data and pseudo annotations are not bundled
in this source release.** Evaluation requires an appropriate trained
checkpoint; the detector initialization alone is not an AlignGroup model.

## Data layout

Obtain the benchmark data from the respective providers and prepare:

```text
COCO/
  train2017/                          # or images2017/train2017/
  val2017/                            # or images2017/val2017/
  filter_1/
    pseudo_train2017_triplet.json
    pseudo_val2017_triplet.json
    meta_data.json
VG/
  VG_100K/
  stanford_filtered/VG-SGG.h5
  stanford_filtered/VG-SGG-dicts.json
  stanford_filtered/image_data.json
  zeroshot_triplet.pytorch
GQA/
  images/
  GQA200-SGG.h5
  GQA200-SGG-dicts.json
  image_data.json
```

`--data_path` in the launcher takes the dataset root shown
above, not its parent. `tools/gqa200_to_h5.py` converts GQA200 to the inherited
VG-style format. The vocabulary/split definitions are in `datasets/vg.py`
and `datasets/vocab_gqa200.py`; do not substitute another split when comparing.

```bash
python tools/check_release.py --coco-root /path/to/COCO --vg-root /path/to/VG
# Optional byte-for-byte comparison against the reference pseudo annotations:
python tools/check_release.py --coco-root /path/to/COCO --verify-coco-hashes
```

Caption parsing and pseudo grounding helpers are in `tools/sg_parser.py` and
`tools/language_sgg_grounding.py`. Optional dependencies are listed in
`requirements-data.txt`. Parsing uses `ALIGNGROUP_CAPTION_FILE` and
`ALIGNGROUP_PARSED_TRIPLETS`; grounding accepts `--img_dir`, `--json_file`
and `--output_dir`. These helpers are not a promise of bitwise regeneration
of the final `filter_1` annotations: parsing models, grounding weights and
filtering/export settings must also match. The reference hashes identify
the exact final files used by the archived experiments.

## Training and evaluation

The public launcher runs **once**, propagates failure, and never kills
processes or automatically restarts. It defaults to validation. Choose GPUs
explicitly; `main_new.py` spawns one worker per visible GPU itself, so do not
wrap these commands in `torchrun`.

```bash
CFG=config/my_config/OvSGTR_Rel_Decoder_Pair_Pruning_V2

# Stage 1: COCO caption-derived pretraining, full AlignGroup.
bash scripts/main_new.sh --gpu 0,1,2,3 --data_type coco_pretrain \
  --config "$CFG/mimic_attn/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_attn_pretrain.py" \
  --data_path /path/to/COCO --out outputs/full_pretrain \
  --pretrain GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --split val --set epochs=8

# Stage 2: use YOUR validation-selected pretraining checkpoint for BOTH roles.
PRETRAIN=/path/to/selected_pretraining/checkpoint.pth
bash scripts/main_new.sh --gpu 0,1,2,3 --data_type vg \
  --config "$CFG/mimic_attn/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_attn_ovdr_finetune.py" \
  --data_path /path/to/VG --out outputs/full_finetune \
  --pretrain "$PRETRAIN" --teacher "$PRETRAIN" --split val

# Final VG test, flat top-k. FINAL is the final fine-tuned checkpoint.
FINAL=/path/to/finetuning/checkpoint0017.pth
bash scripts/main_new.sh --gpu 0,1,2,3 --data_type vg \
  --config "$CFG/mimic_attn/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_attn_ovdr_finetune.py" \
  --data_path /path/to/VG --out outputs/full_test \
  --pretrain "$PRETRAIN" --teacher "$PRETRAIN" \
  --resume "$FINAL" --eval --split test --set rln_eval_v6_topk=True
```

Append `--dry-run` to inspect a command without importing PyTorch or starting
a job. `--set key=value` can be repeated. The original pretraining configs
retain `epochs=18`; the example explicitly limits the budget to eight.
The inherited scheduler is StepLR, not an epoch-budget-dependent cosine
schedule. Fine-tuning keeps its original 18-epoch recipe.

For **CMRA-only**, use the `mimic` pretrain/fine-tune configurations and its
own selected teacher checkpoint. For **SRG-only**, use
`srg_only/SRG_only_pretrain.py` and `srg_only/SRG_only_ovdr_finetune.py` in both
stages. For **GQA**, use `--data_type gqa` and
`mimic_attn/GQA_E6_finetune.py`. OvD, OvR and no-distillation configurations
are retained in the same directories. For a no-distillation config, omit
`--teacher` and use its matching recipe.

Teacher paths in fine-tuning configurations intentionally point to a
nonexistent local placeholder. Always supply `--teacher` with the correct
architecture-matched pretraining checkpoint. Never use a Full checkpoint
as the default SRG-only teacher. `--pretrain` initializes the student;
`--resume` restores a run/checkpoint and is a different operation.

Checkpoint filenames start from zero. `checkpoint0006.pth` is epoch 7;
`checkpoint0017.pth` is epoch 18. Select pretraining checkpoints on VG
**validation**, with the same flat-top-k evaluator, and keep final test
evaluation separate. `rln_eval_v6_topk=False` uses the alternate argmax
evaluation path and is not the reported flat-top-k setting.

## Reference results

VG150 OvD+R SGDet, flat top-k; all values below are multiplied by 100.
The historical rows and the additional SRG-only experiment are separate runs,
as identified in the manuscript. They are reference targets, not results of
a new run of this public packaging.

| Model | Joint R@100 | Novel-object R@100 | Novel-relation R@100 | mR@100 |
| --- | ---: | ---: | ---: | ---: |
| OvSGTR | 16.37 | 17.48 | 11.18 | 3.15 |
| AlignGroup-CMRA | 17.85 | 18.52 | 18.04 | 2.40 |
| SRG-only | 19.31 | 20.05 | 18.49 | 3.07 |
| AlignGroup | 21.04 | 22.79 | 18.22 | 3.60 |

Novel-object recall is relation recall on instances whose subject and object
are both novel; it is not object detection AP. See the paper and supplement
for protocol and checkpoint details. A successful import/source check is
not a reproduction of these metrics.

## Acknowledgements and license

Built on [OvSGTR](https://github.com/gpt4vision/OvSGTR) and
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO).
Upstream licenses and notices are retained; see `LICENSE`,
`GroundingDINO/LICENSE`, and `THIRD_PARTY.md`.

Please cite the AlignGroup manuscript when using these components, together
with the underlying work where appropriate. Publication metadata will be
updated when available.
