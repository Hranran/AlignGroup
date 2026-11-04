"""Convert the GQA200 annotations (SHA-GCL split) into the VG-SGG layout.

Why: `datasets/vg.py` already implements the whole SGG data path (text prompts,
caption building, gt_query slots, distillation, BOX_SCALE recovery).  Rather than
re-implementing it for GQA, we re-shape GQA200 into the exact files `build_vg`
reads, so every downstream line of code stays identical to the VG runs.

Inputs (from https://github.com/dongxingning/SHA-GCL-for-SGG, mirrored by DRM/VETO):
    GQA_200_ID_Info.json   ind_to_classes (201, [0]='__background__')
                           ind_to_predicates (101, [0]='__background__')
    GQA_200_Train.json     filenames_all / img_info_all / gt_boxes_all (xyxy px)
    GQA_200_Test.json      gt_classes_all / relationships_all ([sub, obj, pred])

Outputs (written next to the inputs):
    GQA200-SGG.h5          split / labels / boxes_1024 / relationships /
                           predicates / img_to_{first,last}_{box,rel}
    GQA200-SGG-dicts.json  label_to_idx / predicate_to_idx
    image_data.json        [{image_id, width, height}, ...], same order as the h5

Usage:
    python3 tools/gqa200_to_h5.py --gqa-root /workspace/datasets/GQA
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

# Must match datasets/vg.py
BOX_SCALE = 1024

# Images are laid out train-first so that vg.py's `image_index[:5000]` (val) and
# `image_index[5000:]` (train) reproduce GQA200's own "first 5k of train is val".
TRAIN_FLAG, TEST_FLAG = 0, 2


def load_split(path):
    with open(path, "r") as fin:
        data = json.load(fin)
    keys = ["filenames_all", "img_info_all", "gt_boxes_all",
            "gt_classes_all", "relationships_all"]
    n = len(data[keys[0]])
    for k in keys:
        assert k in data, "%s is missing key %s" % (path, k)
        assert len(data[k]) == n, "%s: len(%s) != len(%s)" % (path, k, keys[0])
    return [data[k] for k in keys], n


def xyxy_to_boxes1024(boxes_xyxy, width, height, stats):
    """xyxy pixels -> (cx, cy, w, h) scaled by BOX_SCALE / max(w, h).

    vg.py recovers boxes with `box / BOX_SCALE * max(w, h)` (vg.py:394), so the
    scale factor here must be that same `max(w, h)` -- not a per-axis scale.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    scale = float(BOX_SCALE) / max(width, height)

    x1 = np.clip(boxes[:, 0], 0, width)
    y1 = np.clip(boxes[:, 1], 0, height)
    x2 = np.clip(boxes[:, 2], 0, width)
    y2 = np.clip(boxes[:, 3], 0, height)

    # vg.py asserts w > 0 and h > 0 on the raw h5 (vg.py:1022); a degenerate box
    # would abort loading, so widen it to one pixel instead of dropping it
    # (dropping would require re-indexing every relation of that image).
    w = x2 - x1
    h = y2 - y1
    stats["degenerate"] += int(((w <= 0) | (h <= 0)).sum())
    x2 = np.where(w <= 0, np.minimum(x1 + 1.0, width), x2)
    y2 = np.where(h <= 0, np.minimum(y1 + 1.0, height), y2)
    # a 1px image edge can still leave w == 0 after the clip above
    x1 = np.where(x2 - x1 <= 0, np.maximum(x2 - 1.0, 0.0), x1)
    y1 = np.where(y2 - y1 <= 0, np.maximum(y2 - 1.0, 0.0), y1)

    out = np.empty_like(boxes)
    out[:, 0] = (x1 + x2) / 2.0 * scale
    out[:, 1] = (y1 + y2) / 2.0 * scale
    out[:, 2] = (x2 - x1) * scale
    out[:, 3] = (y2 - y1) * scale

    # round-trip check against what vg.py will reconstruct
    inv = float(max(width, height)) / BOX_SCALE
    rx1 = (out[:, 0] - out[:, 2] / 2.0) * inv
    ry1 = (out[:, 1] - out[:, 3] / 2.0) * inv
    rx2 = rx1 + out[:, 2] * inv
    ry2 = ry1 + out[:, 3] * inv
    err = np.abs(np.stack([rx1 - x1, ry1 - y1, rx2 - x2, ry2 - y2])).max() if len(boxes) else 0.0
    stats["max_roundtrip_err"] = max(stats["max_roundtrip_err"], float(err))

    return out.astype(np.float32)


def build(gqa_root, out_h5, out_dicts, out_image_data):
    info_path = os.path.join(gqa_root, "GQA_200_ID_Info.json")
    with open(info_path, "r") as fin:
        info = json.load(fin)
    ind_to_classes = info["ind_to_classes"]
    ind_to_predicates = info["ind_to_predicates"]
    assert ind_to_classes[0] == "__background__", ind_to_classes[0]
    assert ind_to_predicates[0] == "__background__", ind_to_predicates[0]
    num_obj_cat = len(ind_to_classes) - 1
    num_rel_cat = len(ind_to_predicates) - 1
    print("[vocab] %d object categories, %d predicate categories"
          % (num_obj_cat, num_rel_cat))

    splits = []
    for fname, flag in [("GQA_200_Train.json", TRAIN_FLAG),
                        ("GQA_200_Test.json", TEST_FLAG)]:
        arrays, n = load_split(os.path.join(gqa_root, fname))
        print("[read ] %-20s %6d images" % (fname, n))
        splits.append((flag, arrays))

    image_data = []
    split_flags = []
    im_first_box, im_last_box, im_first_rel, im_last_rel = [], [], [], []
    all_boxes, all_labels, all_rels, all_preds = [], [], [], []
    box_cursor = rel_cursor = 0
    stats = {"degenerate": 0, "max_roundtrip_err": 0.0,
             "empty_rel": 0, "empty_obj": 0, "dropped_rel": 0}
    seen_names = set()

    for flag, (filenames, img_infos, boxes_all, classes_all, rels_all) in splits:
        for fname, im_info, boxes, classes, rels in zip(
                filenames, img_infos, boxes_all, classes_all, rels_all):
            assert fname not in seen_names, "duplicate image %s" % fname
            seen_names.add(fname)

            stem = os.path.splitext(fname)[0]
            assert stem.isdigit(), "non-numeric GQA filename: %s" % fname
            w, h = int(im_info["width"]), int(im_info["height"])
            image_data.append({"image_id": int(stem), "width": w, "height": h})
            split_flags.append(flag)

            n_obj = len(classes)
            assert len(boxes) == n_obj, "%s: %d boxes vs %d classes" % (fname, len(boxes), n_obj)
            if n_obj == 0:
                stats["empty_obj"] += 1
                im_first_box.append(-1)
                im_last_box.append(-1)
                im_first_rel.append(-1)
                im_last_rel.append(-1)
                continue

            labels = np.asarray(classes, dtype=np.int64)
            assert labels.min() >= 1 and labels.max() <= num_obj_cat, \
                "%s: class index out of range [1, %d]" % (fname, num_obj_cat)
            all_labels.append(labels)
            all_boxes.append(xyxy_to_boxes1024(boxes, w, h, stats))

            im_first_box.append(box_cursor)
            im_last_box.append(box_cursor + n_obj - 1)

            kept = []
            for sub, obj, pred in rels:
                # relations pointing outside this image's object list would
                # silently corrupt the graph -- drop and count them instead
                if not (0 <= sub < n_obj and 0 <= obj < n_obj):
                    stats["dropped_rel"] += 1
                    continue
                assert 1 <= pred <= num_rel_cat, \
                    "%s: predicate index %d out of range" % (fname, pred)
                kept.append((sub + box_cursor, obj + box_cursor, pred))

            if kept:
                arr = np.asarray(kept, dtype=np.int64)
                all_rels.append(arr[:, :2])
                all_preds.append(arr[:, 2])
                im_first_rel.append(rel_cursor)
                im_last_rel.append(rel_cursor + len(kept) - 1)
                rel_cursor += len(kept)
            else:
                stats["empty_rel"] += 1
                im_first_rel.append(-1)
                im_last_rel.append(-1)

            box_cursor += n_obj

    boxes_1024 = np.concatenate(all_boxes, axis=0)
    labels = np.concatenate(all_labels, axis=0).reshape(-1, 1)
    relationships = np.concatenate(all_rels, axis=0)
    predicates = np.concatenate(all_preds, axis=0).reshape(-1, 1)

    # the exact assertions vg.py:1021-1022 will run on this file
    assert np.all(boxes_1024[:, :2] >= 0), "negative box centre survived"
    assert np.all(boxes_1024[:, 2:] > 0), "zero-area box survived"
    assert relationships.max() < len(labels), "relation points past the box table"

    with h5py.File(out_h5, "w") as fout:
        fout.create_dataset("split", data=np.asarray(split_flags, dtype=np.int64))
        fout.create_dataset("labels", data=labels)
        fout.create_dataset("boxes_%d" % BOX_SCALE, data=boxes_1024)
        fout.create_dataset("relationships", data=relationships)
        fout.create_dataset("predicates", data=predicates)
        fout.create_dataset("img_to_first_box", data=np.asarray(im_first_box, dtype=np.int64))
        fout.create_dataset("img_to_last_box", data=np.asarray(im_last_box, dtype=np.int64))
        fout.create_dataset("img_to_first_rel", data=np.asarray(im_first_rel, dtype=np.int64))
        fout.create_dataset("img_to_last_rel", data=np.asarray(im_last_rel, dtype=np.int64))

    # load_info() forces '__background__' to 0 and sorts by index, so writing the
    # 200/100 real categories at 1..N reproduces ind_to_classes exactly
    with open(out_dicts, "w") as fout:
        json.dump({
            "label_to_idx": {n: i for i, n in enumerate(ind_to_classes) if i > 0},
            "predicate_to_idx": {n: i for i, n in enumerate(ind_to_predicates) if i > 0},
        }, fout)

    with open(out_image_data, "w") as fout:
        json.dump(image_data, fout)

    n_train = int(np.sum(np.asarray(split_flags) == TRAIN_FLAG))
    n_test = int(np.sum(np.asarray(split_flags) == TEST_FLAG))
    print("[write] %s" % out_h5)
    print("        images   %d (train %d / test %d)" % (len(split_flags), n_train, n_test))
    print("        objects  %d" % len(labels))
    print("        relations%d" % len(relationships))
    print("[check] max box round-trip error : %.4f px" % stats["max_roundtrip_err"])
    print("[check] widened degenerate boxes : %d" % stats["degenerate"])
    print("[check] images with no relation  : %d" % stats["empty_rel"])
    print("[check] images with no object    : %d" % stats["empty_obj"])
    print("[check] out-of-range relations   : %d" % stats["dropped_rel"])
    return ind_to_classes, ind_to_predicates


def verify(out_h5, out_dicts, out_image_data, ind_to_classes, ind_to_predicates, n_samples=3):
    """Re-read the outputs the way load_graphs() does and print real triplets."""
    print("\n[verify] replaying vg.py's load_graphs() decode path")
    with open(out_image_data, "r") as fin:
        image_data = json.load(fin)
    with h5py.File(out_h5, "r") as fin:
        split = fin["split"][:]
        boxes = fin["boxes_%d" % BOX_SCALE][:]
        labels = fin["labels"][:, 0]
        rels = fin["relationships"][:]
        preds = fin["predicates"][:, 0]
        f_box, l_box = fin["img_to_first_box"][:], fin["img_to_last_box"][:]
        f_rel, l_rel = fin["img_to_first_rel"][:], fin["img_to_last_rel"][:]

    # vg.py:1024-1026 -- cx,cy,w,h -> x1,y1,x2,y2
    boxes = boxes.copy()
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2.0
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]

    for flag, name in [(TRAIN_FLAG, "train"), (TEST_FLAG, "test")]:
        idx = np.where((split == flag) & (f_box >= 0) & (f_rel >= 0))[0]
        print("  %s pool: %d images (vg.py's val = first 5000, train = rest)"
              % (name, len(idx)))
        for i in idx[:n_samples]:
            w = image_data[i]["width"]
            h = image_data[i]["height"]
            # vg.py:394 -- recover pixels
            px = boxes[f_box[i]:l_box[i] + 1] / BOX_SCALE * max(w, h)
            obj_idx = rels[f_rel[i]:l_rel[i] + 1] - f_box[i]
            p = preds[f_rel[i]:l_rel[i] + 1]
            triplets = ["%s -%s-> %s" % (ind_to_classes[labels[f_box[i] + s]],
                                         ind_to_predicates[q],
                                         ind_to_classes[labels[f_box[i] + o]])
                        for (s, o), q in list(zip(obj_idx, p))[:4]]
            print("    %d.jpg (%dx%d, %d objs): %s"
                  % (image_data[i]["image_id"], w, h, len(px), " | ".join(triplets)))
            print("      first box xyxy = [%.1f, %.1f, %.1f, %.1f]"
                  % tuple(px[0].tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gqa-root", required=True,
                    help="directory holding GQA_200_*.json (outputs land here too)")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (default: --gqa-root)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.gqa_root
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, "GQA200-SGG.h5")
    out_dicts = os.path.join(out_dir, "GQA200-SGG-dicts.json")
    out_image_data = os.path.join(out_dir, "image_data.json")

    for path in (out_h5, out_dicts, out_image_data):
        if os.path.exists(path):
            print("[warn ] overwriting %s" % path)

    ind_to_classes, ind_to_predicates = build(
        args.gqa_root, out_h5, out_dicts, out_image_data)
    verify(out_h5, out_dicts, out_image_data, ind_to_classes, ind_to_predicates)
    print("\n[done ] wrote 3 files to %s" % out_dir)


if __name__ == "__main__":
    sys.exit(main())
