# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import copy
import re

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import torch.distributed as dist 

from util.get_param_dicts import get_param_dict, get_param_dict_new
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate_new, train_one_epoch_new

import wandb
from collections import OrderedDict

from groundingdino.models.GroundingDINO import build_groundingdino
# from groundingdino.models.GroundingDINO import build_groundingdino_fuse
# from groundingdino.models.GroundingDINO import build_groundingdino_decoupled
from groundingdino.models.GroundingDINO import build_ovsgtr_base
# from groundingdino.models.GroundingDINO import build_ovsgtr_text_cross
# from groundingdino.models.GroundingDINO import build_ovsgtr_vs3_like
# from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder
# from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder_pair_pruning
from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder_pair_pruning_v2
# from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder_pair_pruning_v3
from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder_pair_pruning_v4
from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder_pair_pruning_v5
from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder_pair_pruning_v6
from groundingdino.models.GroundingDINO import build_ovsgtr_rln_decoder_pair_pruning_v7

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def is_main_process():
    return (not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--anno_train_file', type=str, default='oiv6_train_triple.jsonl')
    parser.add_argument('--anno_val_file', type=str, default='val_bbox_cap.jsonl')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--half_dtype', default="bfloat16", type=str)

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict, "ERROR: modelname:{} not in models:{}".format(args.modelname, MODULE_BUILD_FUNCS._module_dict)

    args.text_encoder_type = os.environ.get('ALIGNGROUP_BERT_PATH', 'bert-base-uncased')

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(gpu, ngpus_per_node, args):
    # ========================== DDP initialize ==========================
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, 'heartbeat.txt'), "w") as f:
            f.write(str(time.time()))

    args.gpu = gpu 
    #utils.init_distributed_mode(args)
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    # ========================== cfg initialize ==========================
    # Load config file and update it to args.
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    if is_distributed():
        rank = dist.get_rank()  # rank of the current process
        device = torch.device(f'cuda:{rank}')  # one GPU per rank
    else:
        device = torch.device('cuda')

    output_dir = Path(args.output_dir)

    if args.distributed:
        dist.barrier()

    # ========================== model initialize ==========================
    # model initialize
    # model without ddp
    # model to cuda
    # load pretrain weight
    # model ddp
    # optimizer initialize
    # resume

    # initialize
    model, criterion, postprocessors = build_model_main(args)
    model_without_ddp = model

    # to device
    model = model.to(device)
    criterion = criterion.to(device)
    postprocessors['bbox'] = postprocessors['bbox'].to(device)

    # load pretrain weights
    if (not args.resume) and args.pretrain_model_path:
        if args.modelname in ['ovsgtr_rln_decoder', 'ovsgtr_rln_decoder_pair_pruning', 'ovsgtr_rln_decoder_pair_pruning_v2', 'ovsgtr_rln_decoder_pair_pruning_v3', 'ovsgtr_rln_decoder_pair_pruning_v4', 'ovsgtr_rln_decoder_pair_pruning_v6', 'ovsgtr_rln_decoder_pair_pruning_v7']:
            logger.info("*"*10 + "Loading weights from pretrained model:%s ..." % args.pretrain_model_path)
            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']

            _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items()})

            missing, unexpected = model_without_ddp.load_state_dict(_tmp_st, strict=False)
            logger.info("Miss keys:{}".format(missing))
            logger.info("Unexpected keys:{}".format(unexpected))

            def end_with(path, check_string):
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                if check_string in name.lower():  # case-insensitive
                    return True
                return False

            if end_with(args.pretrain_model_path, 'groundingdino'):
                rln_text_proj = model_without_ddp.rln_text_proj
                rln_decoder = model_without_ddp.rln_decoder

                feat_map_state = model_without_ddp.feat_map.state_dict()
                transformer_decoder_state = model_without_ddp.transformer.decoder.state_dict()

                missing, unexpected = rln_text_proj.load_state_dict(feat_map_state)
                logger.info("rln_text_proj Miss keys:{}".format(missing))
                logger.info("rln_text_proj Unexpected keys:{}".format(unexpected))
                missing, unexpected = rln_decoder.load_state_dict(transformer_decoder_state, strict=False)
                logger.info("rln_decoder Miss keys:{}".format(missing))
                logger.info("rln_decoder Unexpected keys:{}".format(unexpected))
        else:
            logger.info("*"*10 + "Loading weights from pretrained model:%s ..." % args.pretrain_model_path)
            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']

            _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items()})

            missing, unexpected = model_without_ddp.load_state_dict(_tmp_st, strict=False)
            logger.info("Miss keys:{}".format(missing))
            logger.info("Unexpected keys:{}".format(unexpected))

            rln_text_proj = getattr(model_without_ddp, "rln_text_proj", None)
            rln_text_proj_exist = False 
            for k in _tmp_st.keys():
                if 'rln_text_proj' in k:
                    rln_text_proj_exist = True
                    break

            # If model has rln_text_proj, but pretrained weight has no rln_text_proj, we use feat_map to initialize model's rln_text-proj
            if rln_text_proj is not None and (not args.eval) and not rln_text_proj_exist:
                feat_map_state = model_without_ddp.feat_map.state_dict()
                rln_text_proj.load_state_dict(feat_map_state) # initialize rln_text_proj with feat_map

    if args.teacher_weight:
        logger.info("*"*10 + "use_distill=True!")
        model_t = copy.deepcopy(model)
        teacher_weight = args.teacher_weight
        logger.info("Loading Teacher weight:{}".format(teacher_weight))
        checkpoint = torch.load(teacher_weight, map_location='cpu')['model']
        missing, unexpected = model_t.load_state_dict(utils.clean_state_dict(checkpoint))
        logger.info("Teacher Missing keys:{}".format(missing))
        logger.info("Teacher unexpected keys:{}".format(unexpected))
        model_t.eval()
        model_t = model_t.to(device)
    else:
        model_t = None

    # ddp wrap
    if args.distributed:
        dist.barrier()
        if args.find_unused_params:
            print("*"*10, "Warning: find_unused_parameters = True !")
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    # optimizer
    param_dicts = get_param_dict_new(args, model_without_ddp) if args.modelname in ["ovsgtr_text_cross", "ovsgtr_vs3_like", "ovsgtr_rln_decoder", "ovsgtr_rln_decoder_pair_pruning", "ovsgtr_rln_decoder_pair_pruning_v2", "ovsgtr_rln_decoder_pair_pruning_v3", "ovsgtr_rln_decoder_pair_pruning_v4", "ovsgtr_rln_decoder_pair_pruning_v6", "ovsgtr_rln_decoder_pair_pruning_v7"] else get_param_dict(args, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Load resume weight if exists.
    def find_latest_checkpoint(output_dir):
        """
        Return the full path of the newest checkpointXXXX.pth under
        output_dir, or None if there is none.
        """
        checkpoint_files = []
        pattern = re.compile(r'^checkpoint(\d+)\.pth$')  # matches checkpoint0000.pth
        for fname in os.listdir(output_dir):
            match = pattern.match(fname)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_files.append((epoch_num, fname))
        if not checkpoint_files:
            return None
        # sort by index, take the largest
        latest_epoch, latest_file = max(checkpoint_files, key=lambda x: x[0])
        return os.path.join(output_dir, latest_file)

    if not args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            args.resume = checkpoint_path
        else:
            # auto-resume from the newest epoch checkpoint
            latest_ckpt = find_latest_checkpoint(args.output_dir)
            if latest_ckpt:
                args.resume = latest_ckpt
                print(f"[INFO] No 'checkpoint.pth' found, automatically resuming from latest checkpoint: {latest_ckpt}")
            else:
                print("[INFO] No checkpoint found, training will start from scratch.")
    else:
        print(f"Using given resume path: {args.resume}")

    continue_iter = 0
    if args.resume: # resume
        print(f"Load resume: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing, unexpected = model_without_ddp.load_state_dict(utils.clean_state_dict(checkpoint['model']), strict=False)

        logger.info("RESUME -- Miss keys:{}".format(missing))
        logger.info("RESUME -- Unexpected keys:{}".format(unexpected))
        if 'epoch' in checkpoint:
            logger.info("RESUME -- epoch:{}".format(checkpoint['epoch']))               

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch']
            continue_iter = checkpoint['continue_iter']
            logger.info("RESUME -- continue iter:{}".format(continue_iter))


    # ===== inspect rln-decoder self-attn gates =====
    def print_rln_self_attn_gates(model, tag=""):
        m = model.module if hasattr(model, "module") else model   # DDP / DataParallel
        rln_decoder = getattr(m, "rln_decoder", None)
        if rln_decoder is None:
            print(f"[gate]{tag} no rln_decoder found"); return
        found = False
        for i, layer in enumerate(getattr(rln_decoder, "layers", [])):
            gate = getattr(layer, "self_attn_gate", None)
            if gate is not None:
                found = True
                print(f"[gate]{tag} layer {i}: self_attn_gate = {gate.detach().float().cpu().view(-1).tolist()}")
        if not found:
            print(f"[gate]{tag} no self_attn_gate params (use_self_attn_gate=False?)")

    print_rln_self_attn_gates(model, tag=" after-load")

    # ========================== dataset initialize ==========================
    use_test_set = args.use_test_set
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val' if not use_test_set else 'test', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, drop_last=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train, sampler_val = None, None

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        shuffle=(sampler_train is None),
        collate_fn=utils.collate_fn,
        num_workers=0,
        pin_memory=True
        )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        sampler=sampler_val,
        drop_last=False,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
        pin_memory=True
        )

    # ========================== start training ==========================

    if args.distributed:
        print(f"Rank {get_rank()} ready.")
        dist.barrier()

    if args.eval:
        _ = evaluate_new(model, criterion, postprocessors, data_loader_val, device, output_dir, args=args)
        return
    else:
        # train
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)

            try:
                data_loader_train.dataset.refresh_sample(seed=epoch)
            except:
                print('Dataset cannot been refreshed')

            _ = train_one_epoch_new(
                model=model,
                criterion=criterion,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_max_norm,
                lr_scheduler=lr_scheduler,
                args=args,
                model_t=model_t,
                continue_iter=continue_iter
                )

            lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / f'checkpoint{epoch:04}.pth']
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch + 1,
                        'args': args,
                        'continue_iter': 0,
                    }
                    utils.save_on_master(weights, checkpoint_path)
            continue_iter = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.distributed = ngpus_per_node > 1 # always use ddp for multiple gpus


    if args.distributed:
        args.world_size = args.world_size * ngpus_per_node

        port_id = 10000 + np.random.randint(0, 1000)
        args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(args.rank, ngpus_per_node, args)

