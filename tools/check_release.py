"""Dependency-free source/config/data checks; never starts training or uses a GPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/my_config/OvSGTR_Rel_Decoder_Pair_Pruning_V2'


def load_config(path):
    values = {k: v for k, v in runpy.run_path(str(path)).items()
              if not k.startswith('__')}
    base = values.pop('_base_', None)
    result = {}
    if base:
        for name in ([base] if isinstance(base, str) else base):
            result.update(load_config(path.parent / name))
    result.update(values)
    return result


def require(path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path)
    parser.add_argument('--vg-root', type=Path)
    parser.add_argument('--gqa-root', type=Path)
    parser.add_argument('--bert-path', type=Path)
    parser.add_argument('--verify-coco-hashes', action='store_true')
    args = parser.parse_args()
    files = sorted(ROOT.rglob('*.py'))
    for path in files:
        if any(p in {'.git', '__pycache__', 'build', 'outputs'} for p in path.relative_to(ROOT).parts):
            continue
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for name in ['rln_decoder_transformer.py', 'ovsgtr_rln_decoder_pair_pruning_v2.py']:
        canonical = ROOT / 'GroundingDINO/groundingdino/models/GroundingDINO' / name
        mirror = ROOT / 'models/GroundingDINO' / name
        assert canonical.read_bytes() == mirror.read_bytes(), name + ' copies differ'
    full = load_config(CONFIG / 'mimic_attn/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_attn_pretrain.py')
    cmra = load_config(CONFIG / 'mimic/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_pretrain.py')
    srg = load_config(CONFIG / 'srg_only/SRG_only_pretrain.py')
    assert full['rln_decoder_do_self_attention']
    assert full['rln_decoder_use_text_cross_attention']
    assert full['rln_mask_mode'] == 'subject_grouped'
    assert full['rln_decoder_self_attn_gate_init'] == 1.0
    assert not cmra['rln_decoder_do_self_attention']
    assert srg['rln_decoder_do_self_attention']
    assert not srg['rln_decoder_use_text_cross_attention']
    assert not srg['rln_decoder_use_image_cross_attention']
    fine = load_config(CONFIG / 'mimic_attn/OvSGTR_Rel_Decoder_Pair_Pruning_V2_mimic_attn_ovdr_finetune.py')
    assert fine['rln_eval_v6_topk'] is True
    assert fine['epochs'] == 18
    if args.coco_root:
        root = args.coco_root
        for subset in ['train2017', 'val2017']:
            image_dir = root / subset
            if not image_dir.is_dir():
                image_dir = root / 'images2017' / subset
            require(image_dir)
            require(root / 'filter_1' / ('pseudo_' + subset + '_triplet.json'))
        require(root / 'filter_1/meta_data.json')
        if args.verify_coco_hashes:
            expected = {
                'pseudo_train2017_triplet.json': 'd81b2d5eab4b4f2bae4e0717e1ae9f235cd95fd564a158970af8bc49e89ba5cb',
                'pseudo_val2017_triplet.json': 'f0286675707737c2d0f20cf16c4be3b1fa662b663a20ffbe9289481a53fdbf4d',
                'meta_data.json': '2884b180099db11f9622c13c9785839f7eda7f467e17ccee1e27d12264de18b0',
            }
            for name, reference in expected.items():
                digest = hashlib.sha256()
                with (root / 'filter_1' / name).open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        digest.update(block)
                assert digest.hexdigest() == reference, name + ' differs from reference'
    if args.vg_root:
        for name in ['VG_100K', 'stanford_filtered/VG-SGG.h5',
                     'stanford_filtered/VG-SGG-dicts.json', 'stanford_filtered/image_data.json']:
            require(args.vg_root / name)
    if args.gqa_root:
        for name in ['images', 'GQA200-SGG.h5', 'GQA200-SGG-dicts.json', 'image_data.json']:
            require(args.gqa_root / name)
    if args.bert_path:
        for name in ['config.json', 'vocab.txt']:
            require(args.bert_path / name)
        assert any((args.bert_path / p).is_file() for p in
                   ['pytorch_model.bin', 'model.safetensors']), 'BERT model weights missing'
    print(json.dumps({'source_and_config_checks': 'passed',
                      'full_training_or_metric_verification': False}, indent=2))


if __name__ == '__main__':
    main()
