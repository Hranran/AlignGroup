#!/usr/bin/env bash
# Public launcher. Run one job, propagate its status, and never inspect/kill other jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/GroundingDINO${PYTHONPATH:+:$PYTHONPATH}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DATA_TYPE=""; CONFIG=""; DATA_PATH=""; OUT=""; PRETRAIN=""
RESUME=""; TEACHER=""; SPLIT="val"; EVAL_MODE=0; DRY_RUN=0
OVERRIDES=()
usage() {
    cat <<'EOF'
Usage: bash scripts/main_new.sh --data_type TYPE --config FILE --data_path DIR
       --out DIR --pretrain FILE [--gpu 0,1,2,3] [--teacher FILE]
       [--resume FILE] [--eval] [--split val|test] [--set key=value] [--dry-run]
TYPE: coco_pretrain, vg, or gqa. --set can be repeated.
This launcher runs once; it does not restart jobs or kill GPU processes.
EOF
}
while (( $# )); do
    case "$1" in
        --data_type|--config|--data_path|--out|--pretrain|--gpu|--teacher|--resume|--split|--set)
            if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --data_type) DATA_TYPE="$2";; --config) CONFIG="$2";;
                --data_path) DATA_PATH="$2";; --out) OUT="$2";;
                --pretrain) PRETRAIN="$2";; --gpu) GPU_IDS="$2";;
                --teacher) TEACHER="$2";; --resume) RESUME="$2";;
                --split) SPLIT="$2";; --set) OVERRIDES+=("$2");;
            esac
            shift 2;;
        --eval) EVAL_MODE=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --help|-h) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
    esac
done
if [[ -z "$DATA_TYPE" || -z "$CONFIG" || -z "$DATA_PATH" || -z "$OUT" || -z "$PRETRAIN" ]]; then
    usage >&2; exit 2
fi
case "$DATA_TYPE" in coco_pretrain|vg|gqa) ;; *) echo "Invalid dataset type" >&2; exit 2;; esac
case "$SPLIT" in val) TEST_SET=False;; test) TEST_SET=True;; *) echo "Use --split val or test" >&2; exit 2;; esac
CMD=("${PYTHON:-python}" main_new.py --dataset_file "$DATA_TYPE" -c "$CONFIG"
     --data_path "$DATA_PATH" --output_dir "$OUT" --pretrain_model_path "$PRETRAIN")
[[ -z "$RESUME" ]] || CMD+=(--resume "$RESUME")
(( EVAL_MODE == 0 )) || CMD+=(--eval)
OPTIONS=(dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0
         use_ema=False dn_box_noise_scale=1.0 "use_test_set=$TEST_SET")
[[ -z "$TEACHER" ]] || OPTIONS+=("teacher_weight=$TEACHER")
CMD+=(--amp --options "${OPTIONS[@]}" "${OVERRIDES[@]}")
if (( DRY_RUN )); then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec "${CMD[@]}"
