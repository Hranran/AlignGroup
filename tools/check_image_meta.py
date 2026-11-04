"""Check image_data.json's width/height against the actual files on disk.

vg.py warns ("ERROR index ...") when they disagree, but a warning is not a
verdict. The box round trip is

    box_1024  = box_pixel * BOX_SCALE / max(w, h)      (tools/gqa200_to_h5.py)
    box_pixel = box_1024  * max(w, h) / BOX_SCALE      (datasets/vg.py)

and both directions read the same max(w, h) from image_data.json, so max(w, h)
cancels: a wrong value cannot by itself move a box. (An earlier version of this
script claimed otherwise and reported the wrong statistic.)

The damage comes from the clip in gqa200_to_h5.py instead:

    x1 = clip(x1, 0, width);  y1 = clip(y1, 0, height)

which uses the recorded dimensions. So a record hurts only when it is SMALLER
than the real image on some axis -- then annotations beyond the recorded extent
are silently squashed onto the border. A record that is too large is harmless.

    python tools/check_image_meta.py --data-path /workspace/datasets/GQA \\
                                     --images images
    python tools/check_image_meta.py --data-path /workspace/datasets/VG \\
                                     --images VG_100K \\
                                     --image-file stanford_filtered/image_data.json
"""
import argparse
import json
import os
import sys

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--images", default="images",
                    help="image directory relative to --data-path")
    ap.add_argument("--image-file", default="image_data.json",
                    help="metadata json relative to --data-path")
    ap.add_argument("--list", type=int, default=10,
                    help="how many offending entries to print")
    args = ap.parse_args()

    meta_path = os.path.join(args.data_path, args.image_file)
    img_dir = os.path.join(args.data_path, args.images)
    meta = json.load(open(meta_path))
    print("entries in %s : %d" % (args.image_file, len(meta)))

    missing = mismatch = clipped = 0
    worst = []
    for i, rec in enumerate(meta):
        if i % 10000 == 0:
            sys.stderr.write("  ... %d/%d\r" % (i, len(meta)))
        path = os.path.join(img_dir, "%s.jpg" % rec["image_id"])
        try:
            with Image.open(path) as im:      # header only, does not decode
                aw, ah = im.size
        except Exception:
            missing += 1
            continue
        mw, mh = int(rec["width"]), int(rec["height"])
        if (aw, ah) == (mw, mh):
            continue
        mismatch += 1
        # Harmful only if the record under-states the image on some axis:
        # annotations past the recorded extent get clipped to the border.
        if mw < aw or mh < ah:
            clipped += 1
            worst.append((rec["image_id"], (mw, mh), (aw, ah),
                          max(aw - mw, 0), max(ah - mh, 0)))
    sys.stderr.write(" " * 40 + "\r")

    print()
    print("unreadable / missing files        : %d" % missing)
    print("size mismatch (any dimension)     : %d   <- what vg.py warns about" % mismatch)
    print("record SMALLER than image         : %d   <- the only ones that lose annotation" % clipped)
    if clipped:
        print()
        print("  offenders (image_id, recorded, actual, pixels clipped away):")
        worst.sort(key=lambda t: t[3] + t[4], reverse=True)
        for iid, rec_wh, act_wh, dx, dy in worst[:args.list]:
            print("    %-12s recorded %-12s actual %-12s  lost %d px in x, %d px in y"
                  % (iid, "%dx%d" % rec_wh, "%dx%d" % act_wh, dx, dy))
    print()
    if clipped == 0:
        print("VERDICT: no annotation was lost. The vg.py warnings are cosmetic --")
        print("         every record is at least as large as its image.")
    else:
        print("VERDICT: %d images (%.3f%%) had annotations clipped." % (
            clipped, 100.0 * clipped / max(1, len(meta))))
        print("         Weigh against comparability: these errors live in the")
        print("         dataset's own metadata, so prior work inherits them too.")
        print("         Fixing means rewriting image_data.json from the real sizes")
        print("         AND regenerating the h5 -- both sides must agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
