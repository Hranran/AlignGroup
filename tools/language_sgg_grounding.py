#!/usr/bin/env python
# coding: utf-8

# # import 

# In[1]:

from pathlib import Path
import argparse
import os
import sys
import copy
import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model 

import json
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import build_dataset

import groundingdino.datasets.transforms as T


from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from groundingdino.util.vl_utils import build_captions_and_token_span
from torchvision.ops import nms


# import ignite.distributed as idist
from torch.utils.data import DataLoader, DistributedSampler 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


CHAR = '$*$'



# # Model


def get_model(model_card='swinb'):
    if model_card == 'swinb':
        model_config_path = "config/GroundingDINO_SwinB_pretrain.py"
    else:
        model_config_path = "config/GroundingDINO_SwinT_OGC_pretrain.py"

    args = SLConfig.fromfile(model_config_path)
    args.text_encoder_type = os.environ.get("ALIGNGROUP_BERT_PATH", "bert-base-uncased")
    args.do_sgg = False
    args.rln_pretraining = False 
    
    args.device = 'cpu'
    args.max_text_len = 512
    model, criterion, postprocessors = build_model(args)
    if model_card == 'swinb':
        model_checkpoint_path = os.environ.get("ALIGNGROUP_GDINO_SWINB", "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    else:
        model_checkpoint_path = os.environ.get("ALIGNGROUP_GDINO_SWINT", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
    
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        if name != k:
            print("replace %s with %s" %(k, name))
    
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("missing:", missing, " unepxected:", unexpected)
    
    model = model.to(args.device)
    model.eval()
    
    print("model:", model)

    return model

# In[ ]:

"""
dataset_name = "vg"
args.dataset_file = dataset_name
if dataset_name == 'coco':
    args.data_path = "./data/coco" # the path of coco

elif dataset_name == 'oicap':
    args.data_path = './data/open-imagev6'
    args.anno_train_file = 'oiv6_train_triple.jsonl'
    args.anno_val_file = 'oiv6_val_triple.jsonl'
    
elif dataset_name == "vg":
    args.data_path = "./data/visual_genome"
        
args.fix_size = True
 

train_ds = build_dataset(image_set='val', args=args)   
print("len(data):", len(train_ds))
"""


# In[ ]:

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, json_file, is_coco=False):
        super().__init__()
        self.img_dir = img_dir
        self.is_coco = is_coco

        with open(json_file, 'r') as fin:
            self.data = json.load(fin)

        self.transform = T.Compose(
        [
            #T.RandomResize([800], max_size=1333),
            T.RandomResize([600], max_size=1000),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        )
        # self.repeat = not isinstance(self.data[0]['caption'], str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        name = str(item['image_id']).zfill(12) if self.is_coco else str(item['image_id']) 
        image_path = os.path.join(self.img_dir, name + '.jpg')

        # load image
        try:
            image_pil = Image.open(image_path).convert("RGB")  # load image
        except:
            print("*"*10, "Fail to open:", image_path)
            return self[index - random.randint(1, 10)]

        image, _ = self.transform(image_pil, None)  # 3, h, w


        all_cap, all_tok = [], []
        all_id = []
        # if not self.repeat:
        #     item['relations'] = [item['relations']]
        item['relations'] = [item['relations']]

        names = []
        for ii, all_rels in enumerate(item['relations']):
            text_prompt = []
            for rel in all_rels:
                text_prompt.append(rel[0].lower())
                text_prompt.append(rel[1].lower()) 
            text_prompt = list(set(text_prompt))
            if len(text_prompt) == 0:
                continue 

            captions, cat2tokenspan = build_captions_and_token_span(text_prompt, True) 
            captions = captions.lower()
            captions = captions.strip()
            if not captions.endswith("."):
                captions = captions + "."

            all_cap.append(captions)
            all_tok.append(list(cat2tokenspan.values()))
            all_id.append(f'%s{CHAR}%s'%(item['image_id'], ii))
            names.append(name + CHAR + str(ii) + '.jpg')

        if os.environ.get("DEBUG") != '1':
            image_pil = None

        return image, all_cap, all_tok, all_id, image_pil, names, item

from util.misc import nested_tensor_from_tensor_list, all_gather

def collate_fn(batch):

    imgs = []
    items = {'captions': [], 'tokenspan': [], 'id':[]}
    origin_items = []

    org = []
    names = []
    for item in batch:
        imgs.extend([item[0] for _ in range(len(item[3]))])

        items['captions'].extend(item[1])
        items['tokenspan'].extend(item[2])
        items['id'].extend(item[3])
        org.extend([copy.deepcopy(item[4]) for _ in range(len(item[3]))])
        names.extend(item[5])
        origin_items.append(item[6])

    imgs = nested_tensor_from_tensor_list(imgs)

    return (imgs, items, org, names, origin_items)

from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

def box_cxcywh_to_xyxy(x):  
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), 
    (x_c + 0.5 * w), (y_c + 0.5 * h)] 
    
    return torch.stack(b, dim=-1) 


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        #label = label.split(CHAR)[0]
        #if label == 'girl':
        #    color = (44/100*255, 59/100*255, 47/100*255)
        #elif label == 'umbrella':
        #    color = (54/100*255, 42/100*255, 82/100*255)
        #else:
        #    color = (52.8/100*255, 36.8/100*255, 30./100*255)

        #color = (int(color[0]), int(color[1]), int(color[2]))

        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def get_args_parser():
    parser = argparse.ArgumentParser('GDINO pesudo labeling', add_help=False)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    return parser



# def main(local_rank):
def main(gpu, ngpus_per_node, args):
    print(f"gpu: {gpu}")
    args.gpu = gpu 
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)

    train_ds = ImgDataset(args.img_dir, args.json_file, is_coco=True)
    if args.distributed:
        sampler_train = DistributedSampler(train_ds, drop_last=False)
    else:
        sampler_train = None
    train_loader = DataLoader(train_ds, 
                              batch_size=8,
                              sampler=sampler_train,
                              collate_fn=collate_fn,
                              num_workers=8,
                              shuffle=False)


    all_data = {str(e['image_id']): e for e in train_loader.dataset.data}
    DST = []
    
    model = get_model()
    model = model.cuda(args.gpu)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
    tokenizer = model.module.tokenizer if args.distributed else model.tokenizer
    
    for kk, (images, item, orgs, names, origin_items) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f'rank {dist.get_rank()} pseudo labeling:', position=dist.get_rank())):
        captions = item['captions']
        token_spans = item['tokenspan']
        images = images.to('cuda')
        with torch.no_grad():
            outputs = model(images, captions=captions)
    
        logits = outputs["pred_logits"].sigmoid()  # (bsz, nq, 256)
        boxes = outputs["pred_boxes"]  # (bsz, nq, 4)
    
        box_threshold = 0.25
        
        for bsz in range(logits.shape[0]):
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                tokenizer(captions[bsz]),
                token_span=token_spans[bsz],
                max_text_len=512
            ).to(logits.device) # bsz, n_phrase, 256
    
            logits_for_phrases = positive_maps @ logits[bsz].T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            all_scores = []
    
            for (token_span, logit_phr) in zip(token_spans[bsz], logits_for_phrases):
                # get phrase
                phrase = ' '.join([captions[bsz][_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr >= box_threshold
                # filt box
                all_boxes.append(boxes[bsz][filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if True:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"{CHAR}{str(logit.item())[:4]}" for logit in logit_phr_num])
                    all_scores.extend([logit.item() for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
                    
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases
            boxes_scores = torch.as_tensor(all_scores)
    
            scores = {}
            for phrase in pred_phrases:
                tmp = phrase.split(CHAR)
                if tmp[0] not in scores:
                    scores[tmp[0]] = phrase
                
                old_score = scores[tmp[0]].split(CHAR)[1] 
                if float(tmp[1]) > float(old_score):
                    scores[tmp[0]] = phrase 
            
            new_boxes, new_phrases = [], []
            new_boxes_score = []
            recorded = {}
            for box, box_score, phrase in zip(boxes_filt, boxes_scores, pred_phrases):
                tmp = phrase.split(CHAR)[0]
    
                if phrase == scores[tmp]:
                    recorded[tmp] = True
                    new_boxes.append(box)
                    new_boxes_score.append(box_score.item())
                    new_phrases.append(phrase)
                    
                
            if len(new_boxes) == 0:
                continue 
            
            new_boxes = torch.stack(new_boxes)
            new_boxes_score = torch.as_tensor(new_boxes_score)
            boxes_filt, pred_phrases = new_boxes, new_phrases 
            boxes_scores = new_boxes_score
            keep = nms(boxes_filt, boxes_scores, 0.5)
            boxes_filt = boxes_filt[keep]
            boxes_scores = boxes_scores[keep]
            pred_phrases = [pred_phrases[k] for k in keep]
            
            ## visualize pred
            if os.environ.get("DEBUG") == '1':
                size = orgs[bsz].size
                pred_dict = {
                    "boxes": boxes_filt,
                    "size": [size[1], size[0]],  # H,W
                    "labels": pred_phrases,
                }
            
            keep_phrases = [e.split(CHAR)[0] for e in pred_phrases]
            scale_fct = 1.0 #torch.as_tensor([size[0], size[1], size[0], size[1]])
            pseudo_gt = {'boxes': (box_cxcywh_to_xyxy(boxes_filt) * scale_fct).numpy().round(5).tolist(),
                         'scores': boxes_scores.numpy().tolist(), 
                         'phrases':  keep_phrases
                         }

            if len(keep_phrases) == 0:
                print("no keep phrases")
                continue 
            
            name, idx = item['id'][bsz].split(CHAR)
            # all_rels = all_data[name]['relations'][int(idx)]
    
            new_rels = []
            for total_rel in origin_items[bsz]['relations']:
                for rel in total_rel:
                    if rel[0] in keep_phrases and rel[1] in keep_phrases:
                        new_rels.append(rel)

            if len(new_rels) <= 0:
                # print("no new relations")
                continue

            # idx_ = int(idx)
            # if name not in DST:
            #     DST[name] = copy.deepcopy(all_data[name])

            # DST[name]['relations'][idx_] = new_rels
            # DST[name]['rank'] = idist.get_rank()
            # if 'boxes' not in DST[name]:
            #     len_ = len(DST[name]['relations'])
            #     DST[name]['boxes'] = [[] for _ in range(len_)]
            #     DST[name]['scores'] = [[] for _ in range(len_)]
            #     DST[name]['phrases'] = [[] for _ in range(len_)]

            # DST[name]['boxes'][idx_] = pseudo_gt['boxes']
            # DST[name]['scores'][idx_]  = pseudo_gt['scores']
            # DST[name]['phrases'][idx_] = pseudo_gt['phrases']

            i_dst = {}
            i_dst['image_id'] = origin_items[bsz]['image_id']
            i_dst['id'] = origin_items[bsz]['id']
            i_dst['caption'] = origin_items[bsz]['caption']
            i_dst['relations'] = new_rels
            i_dst['boxes'] = pseudo_gt['boxes']
            i_dst['scores'] = pseudo_gt['scores']
            i_dst['lables'] = pseudo_gt['phrases']

            DST.append(i_dst)

            if kk % 100 == 0:
                print('*'*10, i_dst)
            
            if os.environ.get("DEBUG") == '1':
                image_with_box = plot_boxes_to_image(orgs[bsz], pred_dict)[0]

                plt.clf()
                plt.figure(figsize = (12, 12))
                plt.imshow(image_with_box)
                plt.savefig(str(Path(args.output_dir) / names[bsz]))

            # print(f'DEBUG: {os.environ.get("DEBUG")}')

            if kk > 2 and os.environ.get("DEBUG") == '1':
                break 
            if False: #os.environ.get("DEBUG") == '1':
                plt.clf()
                plt.figure(figsize = (12, 12))
                plt.imshow(image_with_box) 
                plt.savefig("img.jpg")
                import pdb; pdb.set_trace()

        
    # dst_list = all_gather(list(DST.values()))
    # out = []
    # for dst in dst_list:
    #     out.extend(dst)

    world_size = dist.get_world_size()

    gathered = [None] * world_size

    dist.all_gather_object(gathered, DST)

    # concatenate
    merged = []
    for x in gathered:
        merged.extend(x)


    torch.distributed.barrier() 
    if dist.get_rank() == 0:

        with open(args.output_dir, 'w') as fout:
            json.dump(merged, fout, indent=4)

    print("job finished.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GDINO pesudo labeling', parents=[get_args_parser()])
    args = parser.parse_args()

    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.distributed = ngpus_per_node > 1 # always use ddp for multiple gpus

    if args.distributed:
        args.world_size = ngpus_per_node

        port_id = 10000 + np.random.randint(0, 1000)
        args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(args.rank, ngpus_per_node, args)
