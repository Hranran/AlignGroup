# modified from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/data/datasets/visual_genome.py


import os
import sys
import torch
import h5py
import json
import copy
from PIL import Image
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from tqdm import tqdm
import random

from util.bounding_box import BoxList, boxlist_iou
from util.box_ops import box_iou

from datasets.coco import make_coco_transforms
from pycocotools.coco import COCO

BOX_SCALE = 1024  # Scale at which we have the boxes


PREDICATE_BG = '[UNK]' # speical token


def convert_keys_to_int(d):
    # a new dict to hold the converted key-value pairs
    new_dict = {}
    
    for key, value in d.items():
        # if the key is a string, try to convert it to an int
        if isinstance(key, str):
            try:
                # convert the string key to an int
                new_key = int(key)
            except ValueError:
                # if the conversion fails, keep the original key
                new_key = key
        else:
            # if the key is already an int, keep it
            new_key = key
        
        # store the new key and value in the new dict
        new_dict[new_key] = value
    
    return new_dict


def preprocess_caption(caption: str) -> str: 
    result = caption.lower().strip() 
    if result.endswith("."):    
        return result
    return result + "." 

class COCOPretrainDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path,
            split,
            img_dir,
            meta_data_file,
            transforms=None,
            filter_duplicate_rels=True,
            filter_non_overlap=True,
            flip_aug=False,
            use_text_labels=False,
            pretrain_data_percentage=None,
            seed=1,
            ):
        """
        Torch dataset (COCO format) for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        self.dataset_name = "coco_pretrain"
        self.data_path = data_path
        self.split = split
        self.flip_aug = flip_aug
        self.img_dir = img_dir
        self.meta_data_file = meta_data_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        self.data_mask = None
        if pretrain_data_percentage:
            self.pretrain_data_percentage = pretrain_data_percentage
            self.seed = seed
            with open(self.data_path) as f:
                data = json.load(f)
                self.data_length = len(data)
            def generate_subset_mask(data_length, seed, ratio):
                """
                Args:
                    data_list (list): the input list
                    seed (int): random seed
                    ratio (float): in (0, 1], the fraction to keep

                Returns:
                    mask (np.ndarray): bool array of shape (len(data_list),)
                """
                assert 0 < ratio <= 1, "ratio must be in (0, 1]"
                n = data_length
                k = int(n * ratio)
                rng = np.random.default_rng(seed)
                selected = rng.choice(n, size=k, replace=False)
                mask = np.zeros(n, dtype=bool)
                mask[selected] = True
                return mask
            self.data_mask = generate_subset_mask(self.data_length, seed, self.pretrain_data_percentage)

        ind_to_classes, ind_to_predicates, _, _ = load_info(meta_data_file) # contiguous 151, 51 containing __background__
        
        self.ind_to_classes = ind_to_classes 
        self.ind_to_predicates = ind_to_predicates

        self.name2classes = OrderedDict({name: idx  for idx, name in self.ind_to_classes.items() if name != '__background__'})
        self.class2name = OrderedDict({v:k for k, v in self.name2classes.items()})

        self.categories = [{'supercategory': 'person', # not used?
                            'id': idx, 
                            'name': self.ind_to_classes[idx]}  
                            for idx in range(len(self.ind_to_classes)) if self.ind_to_classes[idx] != '__background__'
                            ]
        self.ind_to_predicates = ind_to_predicates
        # self.name2predicates = {ind_to_predicates[k]: k for k in range(len(ind_to_predicates))}
        self.name2predicates = {v: k for k, v in ind_to_predicates.items()}

        gt_boxes, gt_classes, relationships = load_graphs(self.data_path, self.filter_non_overlap, self.split, self.data_mask)

        self.gt_classes = gt_classes
        self.relationships = relationships
        self.gt_boxes = gt_boxes
        filenames, img_info = load_image_filenames(img_dir, data_path, self.data_mask) # length equals to split_mask

        assert len(img_info) == len(gt_boxes), " len(img_info) != len(gt_boxes)"

        self.images = []
        keep_im = [1]*len(filenames)
        for idx, (name, im_info) in enumerate(zip(filenames, img_info)):
            im_info.update({'file_name': name})
            im_info.update({'id': im_info['image_id']})
            self.images.append(im_info)

        self.annotations = []
        self.ids = [im['image_id'] for im, keep in zip(self.images, keep_im) if keep]

        images = []
        for idx, boxes, labels, edges in zip(range(len(self.images)), gt_boxes, gt_classes, relationships):
            if keep_im[idx] != 1:
                continue
            item = {'image_id': self.images[idx]['image_id'],
                    'boxes': boxes, #  normalized xyxy
                    'labels': labels,
                    'edges' : edges, 
                    'iscrowd': 0
                    }
            self.annotations.append(item)
            images.append(self.images[idx])

        self.images = images
        self._coco = None

        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once
        ##correct_img_info(self.img_dir, self.image_file)

        self.use_text_labels = use_text_labels
        self.empty_output_flip = False

    @property
    def coco(self):
        if self._coco is None:
            _coco = COCO()
            coco_dicts = dict(
                            images=self.images, 
                            annotations=[],
                            categories=self.categories)
            
            for ann in tqdm(self.annotations):
                for cls, box in zip(ann['labels'], ann['boxes']):
                    item = {
                            'area': (box[3] - box[1]) * (box[2] - box[0]),
                            'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]], # xywh
                            'category_id': cls,
                            'image_id': ann['image_id'], 
                            'id': len(coco_dicts['annotations']),
                            'iscrowd': 0,
                           }                    
                    coco_dicts['annotations'].append(item)

            _coco.dataset = coco_dicts
            _coco.createIndex()
            self._coco = _coco

        return self._coco

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.empty_output_flip:
            img = None
            target = {'index': index}
            return img, target

        item = self.images[index]
        image_id = item["image_id"]
        w, h = item['width'], item['height']
        
        # load image
        img = Image.open(item['file_name']).convert("RGB")
        if img.size[0] != item['width'] or img.size[1] != item['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', 
                  str(item['width']), ' ', str(item['height']), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')

        # load gt
        gt_boxes, gt_classes, gt_edges = self.get_groundtruth(index)  # origin xyxy

        if self.filter_duplicate_rels and self.split == "train":
            # Filter out dupes!
            old_size = gt_edges.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_edges:
                all_rel_sets[(o0, o1)].append(r)
            gt_edges = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            gt_edges = np.array(gt_edges, dtype=np.int32)

        if self.split == "train":
            gt_boxes[:, [1,3]] = gt_boxes[:, [1, 3]].clip(None, h)
            gt_boxes[:, [0,2]] = gt_boxes[:, [0, 2]].clip(None, w)

        if flip_img:
            new_xmin = w - gt_boxes[:,2]
            new_xmax = w - gt_boxes[:,0]
            gt_boxes[:,0] = new_xmin
            gt_boxes[:,2] = new_xmax

            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        
        target = dict()
        target["iscrowd"] = torch.zeros(gt_boxes.shape[0], dtype=torch.float32)
        target["boxes"] = torch.as_tensor(gt_boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(gt_classes, dtype=torch.int64)
        target["edges"] = torch.as_tensor(gt_edges, dtype=torch.int64)
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        # origin xyxy
        if self.transforms is not None:
            img, target = self.transforms(img, target)  # this should adjust both boxes and edges
        # normalized cxcywh

        # text
        if self.use_text_labels:
            def sample_caption(origin_labels, map_dict, sample_num):
                unique_labels = list(set(origin_labels))

                list_of_int = list(map_dict.keys())
                while len(unique_labels) < sample_num:
                    random_element = random.choice(list_of_int)
                    if random_element != 0 and random_element not in unique_labels:
                        unique_labels.append(random_element)
                return unique_labels

            labels = target['labels'].cpu().numpy().tolist()

            gt_sample = [self.ind_to_classes[e] for e in labels]

            triples = []
            nouns = []
            rels = []
            for edge in target['edges']:
                sub = gt_sample[edge[0]]
                obj = gt_sample[edge[1]]

                triple = (sub, 
                          obj,
                          self.ind_to_predicates[edge[2].item()])
                triples.append(triple)
                nouns.append(sub)
                nouns.append(obj)
                rels.append(triple[2])

            target['relations'] = triples
            nouns = list(set(nouns))
            rels = list(set(rels))

            target['gt_names'] = copy.deepcopy(gt_sample)

            # sampled_entity_labels = sample_caption(labels, self.ind_to_classes, 120)
            sampled_entity_labels = sample_caption(labels, self.ind_to_classes, 100)
            if self.split == 'train':
                random.shuffle(sampled_entity_labels)
            sampled_entity_category = [self.ind_to_classes[e] for e in sampled_entity_labels]
            target['caption'] = '. '.join(sampled_entity_category)
            target['caption'] = preprocess_caption(target['caption'])

            # check_cat_list = sampled_entity_category.split(". ")
            # check_cat_list = [x.strip().lower() for x in check_cat_list]
            # assert len(check_cat_list) == 100, print(sampled_entity_category)

            target['gt_rels'] = copy.deepcopy(rels)

            relation_labels = []
            for edge in target['edges']:
                relation_labels.append(edge[2].item())
            sampled_relation_labels = sample_caption(relation_labels, self.ind_to_predicates, 50)
            if self.split == 'train':
                random.shuffle(sampled_relation_labels)
            sampled_relation_category = [self.ind_to_predicates[e] for e in sampled_relation_labels]
            rel_caption = '. '.join(sampled_relation_category) + '.'
            target['rel_caption'] = rel_caption

        if len(target['edges']) == 0 and self.split == 'train':
            return self[index - random.randint(1, 10)] # remap to a valid sample

        # box tag: normalized cxcywh
        return img, target

    def get_groundtruth(self, index):
        item = self.images[index]
        w, h = item['width'], item['height']

        ann = self.annotations[index]
        gt_boxes = ann['boxes']
        gt_classes = ann['labels']
        gt_edges = ann['edges']          

        # important: recover original box from BOX_SCALE
        gt_boxes = gt_boxes * np.array([w, h, w, h])
        gt_boxes = torch.from_numpy(np.asarray(gt_boxes, dtype=np.float32)).reshape(-1, 4)  # guard against no boxes

        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int64)

        # combine entities by with iou>0.9
        all_entities_infos_after_iou_merge = []
        ent_match_info = (box_iou(gt_boxes, gt_boxes)[0] > 0.9 ) & (gt_classes.unsqueeze(1) == gt_classes)

        keep_tgt_ids, ent_id_to_new_id = [], {}
        for ent_id in range(len(ent_match_info)):
            matched_ent_ids = ent_match_info[ent_id].nonzero().squeeze(1).tolist()
            if len(matched_ent_ids) > 0:
                keep_tgt_ids.append(ent_id)
                for id in matched_ent_ids: ent_id_to_new_id[id] = len(all_entities_infos_after_iou_merge)
                #all_entities_infos_after_iou_merge.append(all_entities_infos[ent_id])
                all_entities_infos_after_iou_merge.append(-1)
                ent_match_info[:, matched_ent_ids] = False
        
        gt_boxes = gt_boxes[keep_tgt_ids]
        gt_classes = gt_classes[keep_tgt_ids]
        new_edges = []
        for edge in gt_edges:
            new_edges.append([ent_id_to_new_id[edge[0]], ent_id_to_new_id[edge[1]], edge[2]])
        gt_edges = np.asarray(new_edges)


        return gt_boxes, gt_classes, gt_edges

    def get_statistics(self):
        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file, dict_file=self.meta_data_file,
                                                image_file=self.image_file, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            #'att_classes': self.ind_to_attributes,
        }
        return result    

    def refresh_sample(self, seed=None):
        if seed is None:
            self.seed = self.seed + 1
        else:
            self.seed = seed
        with open(self.data_path) as f:
            data = json.load(f)
            self.data_length = len(data)
        def generate_subset_mask(data_length, seed, ratio):
            """
            Args:
                data_list (list): the input list
                seed (int): random seed
                ratio (float): in (0, 1], the fraction to keep

            Returns:
                mask (np.ndarray): bool array of shape (len(data_list),)
            """
            assert 0 < ratio <= 1, "ratio must be in (0, 1]"
            n = data_length
            k = int(n * ratio)
            rng = np.random.default_rng(seed)
            selected = rng.choice(n, size=k, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[selected] = True
            return mask
        self.data_mask = generate_subset_mask(self.data_length, seed, self.pretrain_data_percentage)

        gt_boxes, gt_classes, relationships = load_graphs(self.data_path, self.filter_non_overlap, self.split, self.data_mask)

        self.gt_classes = gt_classes
        self.relationships = relationships
        self.gt_boxes = gt_boxes
        filenames, img_info = load_image_filenames(self.img_dir, self.data_path, self.data_mask) # length equals to split_mask

        assert len(img_info) == len(gt_boxes), " len(img_info) != len(gt_boxes)"

        self.images = []
        keep_im = [1]*len(filenames)
        for idx, (name, im_info) in enumerate(zip(filenames, img_info)):
            im_info.update({'file_name': name})
            im_info.update({'id': im_info['image_id']})
            self.images.append(im_info)

        self.annotations = []
        self.ids = [im['image_id'] for im, keep in zip(self.images, keep_im) if keep]

        images = []
        for idx, boxes, labels, edges in zip(range(len(self.images)), gt_boxes, gt_classes, relationships):
            if keep_im[idx] != 1:
                continue
            item = {'image_id': self.images[idx]['image_id'],
                    'boxes': boxes, #  normalized xyxy
                    'labels': labels,
                    'edges' : edges, 
                    'iscrowd': 0
                    }
            self.annotations.append(item)
            images.append(self.images[idx])

        self.images = images
        self._coco = None

        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once
        ##correct_img_info(self.img_dir, self.image_file)


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    # make a copy
    with open(image_file.replace(".json", "_bak.json"), "w") as fout:
        json.dump(data, fout)

    tmp = []
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
            tmp.append(img)

    print("replace:", len(tmp))
    with open(image_file, 'w') as outfile:  
        json.dump(data, outfile)

def load_info(dict_file):
    """
    Loads the file containing the coco pretrain meta data.
    """
    with open(dict_file, 'r') as f:
        meta_data = json.load(f)

    ind_to_classes = meta_data['entity_int_to_category']
    ind_to_predicates = meta_data['relation_int_to_category']
    classes_to_ind = meta_data['entity_category_to_int']
    predicates_to_ind = meta_data['relation_category_to_int']

    ind_to_classes['0'] = '__background__'
    ind_to_predicates['0'] = PREDICATE_BG
    classes_to_ind['__background__'] = 0
    predicates_to_ind[PREDICATE_BG] = 0

    ind_to_classes = convert_keys_to_int(ind_to_classes)
    ind_to_predicates = convert_keys_to_int(ind_to_predicates)

    return ind_to_classes, ind_to_predicates, classes_to_ind, predicates_to_ind

def load_image_filenames(img_dir, data_path, mask=None):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    if mask is not None:
        assert len(mask) == len(data)
        data = [x for x, m in zip(data, mask) if m]

    fns = []
    img_info = []
    for i_data in data:
        basename = f"{i_data['image_id']:012d}.jpg"

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            i_img_info = {
                "image_id": i_data['image_id'],
                "width": i_data['image_wh'][0],
                "height": i_data['image_wh'][1]
            }

            img_info.append(i_img_info)
        else:
            print(filename)
            raise

    return fns, img_info

def load_graphs(data_path, filter_non_overlap, split, mask=None):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    if mask is not None:
        assert len(mask) == len(data)
        data = [x for x, m in zip(data, mask) if m]

    boxes = []
    gt_classes = []
    relationships = []
    for i_data in tqdm(data, total=len(data), desc='loading graph'):
        boxes.append(np.array(i_data['boxes']))  # normalized xyxy
        gt_classes.append(np.array(i_data['labels']))

        rels = np.array(i_data['relations'])
        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(np.array(i_data['boxes']), (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            rels = rels[inc]

        relationships.append(rels)

    return boxes, gt_classes, relationships

def build_coco_pretrain(image_set, args, disable_transforms=False):
    # data_path = os.path.join(args.data_path, "visual_genome")
    data_path = args.data_path

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    transforms = make_coco_transforms(image_set, 
        fix_size=getattr(args, "fix_size", False),  
        strong_aug=strong_aug, args=args) if not disable_transforms else None

    use_text_labels = getattr(args, "use_text_labels", False)

    filter_non_overlap = getattr(args, "filter_non_overlap", True)

    # return COCOPretrainDataset(
    #     data_path='/path/to/datasets/coco/my_triplet_annotation/pseudo_train2017_triplet.json' if image_set == 'train' else '/path/to/datasets/coco/my_triplet_annotation/pseudo_val2017_triplet.json',
    #     split=image_set,
    #     img_dir='/path/to/datasets/coco/images2017/train2017' if image_set == 'train' else '/path/to/datasets/coco/images2017/val2017',
    #     meta_data_file='/path/to/datasets/coco/my_triplet_annotation/meta_data.json',
    #     transforms=transforms,
    #     use_text_labels=use_text_labels,
    #     filter_non_overlap=filter_non_overlap,
    #     pretrain_data_percentage=getattr(args, "pretrain_data_percentage", None)
    #     )

    # return COCOPretrainDataset(
    #     data_path='/path/to/datasets/coco/filter_5/pseudo_train2017_triplet.json' if image_set == 'train' else '/path/to/datasets/coco/filter_5/pseudo_val2017_triplet.json',
    #     split=image_set,
    #     img_dir='/path/to/datasets/coco/images2017/train2017' if image_set == 'train' else '/path/to/datasets/coco/images2017/val2017',
    #     meta_data_file='/path/to/datasets/coco/filter_5/meta_data.json',
    #     transforms=transforms,
    #     use_text_labels=use_text_labels,
    #     filter_non_overlap=filter_non_overlap,
    #     pretrain_data_percentage=getattr(args, "pretrain_data_percentage", None)
    #     )

    # Use the supplied COCO root; support both A and B image layouts.
    subset = 'train2017' if image_set == 'train' else 'val2017'
    annotation_dir = os.path.join(data_path, 'filter_1')
    img_dir = os.path.join(data_path, subset)
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(data_path, 'images2017', subset)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError('COCO image directory not found: ' + img_dir)

    return COCOPretrainDataset(
        data_path=os.path.join(annotation_dir, 'pseudo_' + subset + '_triplet.json'),
        split=image_set,
        img_dir=img_dir,
        meta_data_file=os.path.join(annotation_dir, 'meta_data.json'),
        transforms=transforms,
        use_text_labels=use_text_labels,
        filter_non_overlap=filter_non_overlap,
        pretrain_data_percentage=getattr(args, "pretrain_data_percentage", None)
        )

# def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True):
#     train_data = VGDataset(split='train', img_dir=img_dir, roidb_file=roidb_file, 
#                         dict_file=dict_file, image_file=image_file, num_val_im=5000, 
#                         filter_duplicate_rels=False)
#     num_obj_classes = len(train_data.ind_to_classes)
#     num_rel_classes = len(train_data.ind_to_predicates)
#     fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
#     bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

#     for ex_ind in tqdm(range(len(train_data))):
#         gt_classes = train_data.gt_classes[ex_ind].copy()
#         gt_relations = train_data.relationships[ex_ind].copy()
#         gt_boxes = train_data.gt_boxes[ex_ind].copy()

#         # For the foreground, we'll just look at everything
#         o1o2 = gt_classes[gt_relations[:, :2]]
#         for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
#             fg_matrix[o1, o2, gtr] += 1
#         # For the background, get all of the things that overlap.
#         o1o2_total = gt_classes[np.array(
#             box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
#         for (o1, o2) in o1o2_total:
#             bg_matrix[o1, o2] += 1

#     return fg_matrix, bg_matrix    


if __name__ == "__main__":
    import sys
    from tqdm import tqdm 
    from groundingdino.util.slconfig import SLConfig
    #from models.GroundingDINO.utils import FrequencyBias

    cfg_file = sys.argv[1]
    args = SLConfig.fromfile(cfg_file)

    args.data_path = './data' #sys.argv[2]
    args.sg_ovd_mode = False #True
    args.sg_ovr_mode = False #True
    args.filter_non_overlap = False
    args.vg_roidb_key = 'split'

    #args.gpt4sgg_file = "./data/visual_genome/vg150_gpt4sgg.json"
    #args.use_gpt4sgg = True


    # dataset = build_vg("test", args, True)
    # cnt = [0, 0]
    # for item in tqdm(dataset):
    #     import pdb; pdb.set_trace()

