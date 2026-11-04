"""GQA200 vocabulary and its open-vocabulary base/novel split.

The 200 object / 100 predicate vocabulary itself comes from the GQA200 split of
SHA-GCL (Dong et al., CVPR'22) and is carried by ``GQA200-SGG-dicts.json``,
which ``tools/gqa200_to_h5.py`` writes; this module only records which of those
categories are *base* (seen during fine-tuning) and which are *novel*.

The split below is transcribed verbatim from ACC (arXiv:2502.03856) Table S1,
p.22, so that our numbers are directly comparable to theirs. It has been
verified that the union of the two object lists and the union of the two
predicate lists are each set-equal to the vocabulary in SHA-GCL's
``GQA_200_ID_Info.json`` -- the symmetric difference is empty on both sides,
including oddities such as ``hang on`` which do not occur in raw GQA. That
equality is what proves the file we converted is the one ACC used.

NOTE: the order here is Table S1's print order, which is NOT the GQA200 class
index order. Use these lists for set membership only; the canonical index
order always comes from the dicts JSON via ``load_info``.

Unlike VG150, GQA200 has no published "novel -> nearest base" synonym table
(that mapping is an OvSGTR design specific to VG150). ACC p.7 states that novel
categories are simply excluded during fine-tuning, so ``GQA200_NOVEL2BASE`` is
deliberately empty: novel objects are dropped rather than relabelled. This
difference must be stated in the paper.
"""

GQA200_BASE_PREDICATES = [
    "parked on", "growing on", "standing in front of", "wearing", "standing on", "with",
    "looking at", "under", "carrying", "near", "above", "covered in", "behind", "at",
    "using", "hanging from", "sitting on", "flying in", "watching", "covering",
    "mounted on", "in front of", "lying on", "standing next to", "grazing in", "holding",
    "beside", "on the back of", "catching", "running on", "swimming in", "playing on",
    "on top of", "floating in", "talking on", "on the bottom of", "standing behind",
    "leaning against", "covered by", "facing", "filled with", "attached to",
    "sitting next to", "next to", "worn on", "in", "on the side of", "driving",
    "close to", "surrounded by", "lying in", "hitting", "pulling", "swinging",
    "touching", "eating", "throwing", "skiing on", "driving on", "hang on", "riding",
    "playing in", "crossing", "walking with", "on", "growing in", "sitting in",
    "cutting", "feeding", "leaning on",
]  # 70

GQA200_NOVEL_PREDICATES = [
    "on the front of", "reaching for", "flying", "of", "parked along", "talking to",
    "sitting at", "standing by", "hanging on", "covered with", "standing near",
    "full of", "surrounding", "walking in", "reflected in", "walking down",
    "walking on", "contain", "below", "printed on", "driving down", "waiting for",
    "resting on", "playing with", "standing in", "grazing on", "by", "around",
    "pulled by", "beneath",
]  # 30

GQA200_BASE_OBJECTS = [
    "mountain", "cow", "people", "face", "number", "pizza", "tire", "player", "pillow",
    "screen", "truck", "kite", "trunk", "sock", "neck", "glove", "coat", "letter",
    "roof", "windshield", "desk", "paw", "leaf", "flower", "plant", "counter", "paper",
    "eye", "book", "branch", "lamp", "cup", "phone", "toilet", "skateboard", "logo",
    "laptop", "vehicle", "motorcycle", "hill", "curtain", "nose", "sheep", "bowl",
    "wire", "bear", "banana", "mouth", "drawer", "shelf", "cap", "animal", "bottle",
    "box", "airplane", "finger", "room", "flag", "seat", "tower", "wing", "fruit",
    "rock", "house", "pot", "bird", "umbrella", "surfboard", "lady", "tie", "fork",
    "vase", "bag", "orange", "clock", "sidewalk", "food", "sink", "cabinet", "beach",
    "boat", "basket", "helmet", "child", "racket", "post", "guy", "towel", "arm",
    "napkin", "bush", "bench", "person", "cone", "apple", "jacket", "fur", "air",
    "sign", "bus", "wrist", "frame", "floor", "dress", "street", "shoe", "ball",
    "girl", "ear", "boy", "broccoli", "fence", "uniform", "hair", "sneakers",
    "blanket", "zebra", "train", "camera", "sticker", "license plate", "lid", "tomato",
    "pants", "giraffe", "watch", "wall", "leg", "bed", "t-shirt", "shorts", "horse",
    "spots", "arrow", "field", "bread", "bicycle", "knife", "couch", "ceiling",
]  # 140

GQA200_NOVEL_OBJECTS = [
    "ocean", "car", "picture", "hand", "snow", "horn", "woman", "sweater", "container",
    "paint", "feet", "clouds", "foot", "dirt", "faucet", "chair", "sand", "tail",
    "stone", "cat", "tag", "traffic light", "keyboard", "tree", "leaves", "elephant",
    "ground", "glass", "frisbee", "trash can", "word", "man", "jeans", "door",
    "building", "sky", "table", "wheel", "pole", "collar", "hat", "cheese", "mane",
    "shirt", "dog", "cord", "cake", "donut", "plate", "backpack", "mirror",
    "street light", "skis", "window", "grass", "water", "bike", "road", "head",
    "cell phone",
]  # 60

# Novel objects are dropped at training time; see the module docstring.
GQA200_NOVEL2BASE = {}


def _self_check():
    assert len(GQA200_BASE_PREDICATES) == 70, len(GQA200_BASE_PREDICATES)
    assert len(GQA200_NOVEL_PREDICATES) == 30, len(GQA200_NOVEL_PREDICATES)
    assert len(GQA200_BASE_OBJECTS) == 140, len(GQA200_BASE_OBJECTS)
    assert len(GQA200_NOVEL_OBJECTS) == 60, len(GQA200_NOVEL_OBJECTS)
    assert len(set(GQA200_BASE_PREDICATES)) == 70, "duplicate base predicate"
    assert len(set(GQA200_NOVEL_PREDICATES)) == 30, "duplicate novel predicate"
    assert len(set(GQA200_BASE_OBJECTS)) == 140, "duplicate base object"
    assert len(set(GQA200_NOVEL_OBJECTS)) == 60, "duplicate novel object"
    assert not (set(GQA200_BASE_PREDICATES) & set(GQA200_NOVEL_PREDICATES))
    assert not (set(GQA200_BASE_OBJECTS) & set(GQA200_NOVEL_OBJECTS))


_self_check()


if __name__ == "__main__":
    print("GQA200 split ok: %d/%d objects, %d/%d predicates (base/novel)" % (
        len(GQA200_BASE_OBJECTS), len(GQA200_NOVEL_OBJECTS),
        len(GQA200_BASE_PREDICATES), len(GQA200_NOVEL_PREDICATES)))
