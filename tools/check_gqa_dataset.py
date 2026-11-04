"""Pre-flight checks for the GQA200 dataset wiring (gates 2, 3 and the
prompt-length check). Runs on CPU in a couple of minutes; run it before
spending any GPU time.

    python tools/check_gqa_dataset.py --data-path /workspace/datasets/GQA

Gate 2  dataset builds, split sizes are as expected
Gate 3  the ovd/ovr training set contains no novel category
Gate T  the BERT prompt fits in max_text_len (silent truncation would make the
        tail of the vocabulary unrecallable)
Gate V  static proof that the VG150 path is unchanged by the refactor
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.vg import (build_gqa, preprocess_caption,
                         VG150_OBJ_CATEGORIES, VG150_BASE_OBJ_CATEGORIES,
                         VG150_NOVEL2BASE, VG150_PREDICATES, PREDICATE_BG)
from datasets.vocab_gqa200 import (GQA200_BASE_OBJECTS, GQA200_NOVEL_OBJECTS,
                                   GQA200_BASE_PREDICATES, GQA200_NOVEL_PREDICATES)


class _Args(object):
    """Minimal stand-in for the training args namespace."""
    def __init__(self, data_path, ovd, ovr):
        self.data_path = data_path
        self.strong_aug = False
        self.fix_size = False
        self.use_text_labels = True
        self.vg_roidb_key = 'split'
        self.filter_non_overlap = True
        self.sg_ovd_mode = ovd
        self.sg_ovr_mode = ovr
        self.use_distill = False
        self.unsupervised_distill = False
        self.gpt4sgg_file = None
        self.use_gpt4sgg = False


def ok(msg):
    print("  [ok]   " + msg)


def fail(msg):
    print("  [FAIL] " + msg)
    return 1


def gate_v():
    """The refactor replaced novel2base[k] with novel2base.get(k, []).

    For VG150 that is only a no-op if every non-base object category is a key
    of VG150_NOVEL2BASE. Prove it rather than assume it.
    """
    print("\nGate V -- VG150 path is provably unchanged")
    errs = 0
    non_base = set(VG150_OBJ_CATEGORIES) - set(VG150_BASE_OBJ_CATEGORIES)
    missing = non_base - set(VG150_NOVEL2BASE)
    if missing:
        errs += fail(".get() would change VG150 behaviour for: %s" % sorted(missing))
    else:
        ok("every non-base VG150 category is a NOVEL2BASE key "
           "(%d of them) -> .get() is a no-op for VG" % len(non_base))
    return errs


def gate_2_3(data_path):
    print("\nGate 2 -- datasets build, split sizes")
    errs = 0
    sets = {}
    for split, ovd, ovr in (('train', True, True), ('val', False, False), ('test', False, False)):
        ds = build_gqa(split, _Args(data_path, ovd, ovr), disable_transforms=True)
        sets[split] = ds
        print("  %-5s : %6d images | %d obj cats | %d predicates"
              % (split, len(ds), len(ds.ind_to_classes), len(ds.ind_to_predicates)))

    # GQA200: 57,623 train-pool images, first 5,000 held out as val.
    if len(sets['val']) != 5000:
        errs += fail("val should be exactly 5000 images, got %d" % len(sets['val']))
    else:
        ok("val = 5000 (matches GQA200's own convention and vg.py's [:5000])")
    if len(sets['test']) != 8209:
        errs += fail("test should be 8209 images, got %d" % len(sets['test']))
    else:
        ok("test = 8209")
    # train is the remaining 52,623 minus whatever the ovd/ovr base-only filter
    # drops, so it is a bound rather than an equality.
    if not (0 < len(sets['train']) <= 52623):
        errs += fail("train size %d outside (0, 52623]" % len(sets['train']))
    else:
        ok("train = %d (<= 52623 after base-only filtering)" % len(sets['train']))

    print("\nGate 3 -- no novel category leaks into the ovd/ovr training set")
    train = sets['train']
    novel_obj = set(GQA200_NOVEL_OBJECTS)
    novel_pred = set(GQA200_NOVEL_PREDICATES)

    seen_obj, seen_pred = set(), set()
    for labels in train.gt_classes:
        seen_obj.update(int(v) for v in labels)
    for rels in train.relationships:
        seen_pred.update(int(r[2]) for r in rels)
    seen_obj_names = {train.ind_to_classes[i] for i in seen_obj}
    seen_pred_names = {train.ind_to_predicates[i] for i in seen_pred}

    leaked_o = sorted(seen_obj_names & novel_obj)
    leaked_p = sorted(seen_pred_names & novel_pred)
    if leaked_o:
        errs += fail("%d NOVEL OBJECTS present in train: %s" % (len(leaked_o), leaked_o[:10]))
    else:
        ok("0 novel objects in train (%d distinct base objects seen)" % len(seen_obj_names))
    if leaked_p:
        errs += fail("%d NOVEL PREDICATES present in train: %s" % (len(leaked_p), leaked_p[:10]))
    else:
        ok("0 novel predicates in train (%d distinct base predicates seen)" % len(seen_pred_names))

    # The evaluator's novel index lists.
    test = sets['test']
    if test.novel_obj_cats is None or test.novel_rel_cats is None:
        errs += fail("novel_obj_cats/novel_rel_cats are None -- the evaluator "
                     "would silently fall back to the VG150 indices!")
    else:
        n_o, n_p = len(test.novel_obj_cats), len(test.novel_rel_cats)
        if n_o != 60 or n_p != 30:
            errs += fail("expected 60 novel objects / 30 novel predicates, got %d/%d" % (n_o, n_p))
        else:
            ok("evaluator gets 60 novel object ids and 30 novel predicate ids")
        names_o = {test.ind_to_classes[i] for i in test.novel_obj_cats}
        if names_o != novel_obj:
            errs += fail("novel object ids do not match ACC Table S1")
        else:
            ok("novel object ids round-trip to exactly ACC's Table S1 names")
        names_p = {test.ind_to_predicates[i] for i in test.novel_rel_cats}
        if names_p != novel_pred:
            errs += fail("novel predicate ids do not match ACC Table S1")
        else:
            ok("novel predicate ids round-trip to exactly ACC's Table S1 names")
    return errs, sets


def gate_t(sets, max_text_len):
    """Measure the real BERT wordpiece length of the prompts."""
    print("\nGate T -- prompt fits in max_text_len = %d" % max_text_len)
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:            # noqa: BLE001 - diagnostic only
        print("  [skip] could not load bert-base-uncased (%s)" % e)
        print("         RUN THIS ON THE TRAINING MACHINE -- do not start the")
        print("         fine-tune until this gate has actually been measured.")
        return 0

    train, test = sets['train'], sets['test']

    def obj_caption(ds, base_only):
        voc = ds.voc_base_obj_categories if base_only else ds.voc_obj_categories
        return preprocess_caption('. '.join(voc[1:]))

    def rel_caption(ds, base_only):
        voc = ds.voc_base_predicates if base_only else ds.voc_predicates
        return '. '.join(voc[1:]) + '.'

    errs = 0
    for label, ds, base_only in (("train (base only)", train, True),
                                 ("test  (full vocab)", test, False)):
        n_obj = len(tk(obj_caption(ds, base_only))['input_ids'])
        n_rel = len(tk(rel_caption(ds, base_only))['input_ids'])
        # concat_rel_text: each caption goes through BERT on its own (so each
        # must fit BERT's 512 positions), then the two ENCODINGS are
        # concatenated -- and it is that concatenation that ContrastiveEmbed
        # pads to max_text_len. Checking the captions separately, as an earlier
        # version of this script did, misses the constraint that actually bites.
        total = n_obj + n_rel
        print("  %s: object %4d + relation %4d = %4d wordpieces" % (label, n_obj, n_rel, total))
        for what, n, limit in (("object caption vs BERT ", n_obj, 512),
                               ("relation caption vs BERT", n_rel, 512),
                               ("concatenated vs max_text_len", total, max_text_len)):
            if n > limit:
                print("    [OVERFLOW] %s : %d > %d" % (what, n, limit))
                errs = 1
    if errs:
        print("\n  >>> Raise max_text_len in the GQA config above the concatenated")
        print("  >>> length (ContrastiveEmbed has no weights, so this is free), or")
        print("  >>> the vocabulary tail is silently unrecallable. If a single")
        print("  >>> caption exceeds 512, BERT itself cannot encode it and the")
        print("  >>> prompt must be split instead.")
    else:
        print("  [ok]   both captions fit BERT, and their concatenation fits max_text_len")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True,
                    help="directory holding GQA200-SGG.h5, GQA200-SGG-dicts.json, "
                         "image_data.json and images/")
    ap.add_argument("--max-text-len", type=int, default=512)
    args = ap.parse_args()

    errs = gate_v()
    e, sets = gate_2_3(args.data_path)
    errs += e
    errs += gate_t(sets, args.max_text_len)

    print("\n" + "=" * 60)
    if errs:
        print("%d CHECK(S) FAILED -- do not start training." % errs)
    else:
        print("all checks passed.")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
