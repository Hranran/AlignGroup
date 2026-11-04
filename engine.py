# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
from tqdm import tqdm
import traceback
from datetime import datetime
from typing import Optional
import signal
import time

from util.utils import slprint, to_device

import torch
import torch.distributed as dist

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.sgg_eval import SggEvaluator 
from datasets.panoptic_eval import PanopticEvaluator


# timeout handler
def timeout_handler(signum, frame):
    print("[ERROR] Timeout reached. Restarting training.")
    os._exit(1)  # force-exit this process so it can be restarted

# timeout set to 10 minutes (600 seconds)
def set_timeout():
    signal.signal(signal.SIGALRM, timeout_handler)  # install the timeout signal
    signal.alarm(60)  # fire the timeout after one minute


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


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


def print_gpu_memory():
    # current and peak allocated GPU memory
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
    print(f"Allocated memory: {allocated:.2f} GB")
    print(f"Max allocated memory this cycle: {max_allocated:.2f} GB")
    print(f"Reserved memory: {reserved:.2f} GB")

    # reset the peak memory counter
    torch.cuda.reset_max_memory_allocated()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, 
                    logger=None, ema_m=None, wandb_logger=None, 
                    model_t=None):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_update_interval = getattr(args, "teacher_update_interval", 1000)
    momentum_t = getattr(args, "teacher_momentum", 0.999)
    teacher_update = getattr(args, "teacher_update", False)


    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        args.global_iter += 1

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets, global_iter=args.global_iter)
            else:
                outputs = model(samples, global_iter=args.global_iter)
            
            if model_t is not None:
                with torch.no_grad():
                    if need_tgt_for_training:
                        outputs_t = model_t(samples, targets, global_iter=args.global_iter)
                    else:
                        outputs_t = model_t(samples, global_iter=args.global_iter)
            else:
                outputs_t = None 
                    
            loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
                                  global_iter=args.global_iter)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

               

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if model_t is not None and teacher_update and args.global_iter % teacher_update_interval == 0:
            if utils.get_rank() == 0:
                print("*"*10, "global iter:", args.global_iter, 
                  " update teacher's weights with momentum:%s!" % momentum_t)
            for name, p in model_t.named_parameters():
                state = model.state_dict()
                keys = list(state.keys())
                if 'module.' in keys[0]:
                    name = 'module.' + name

                with torch.no_grad():
                    p.data = momentum_t * p.data + (1-momentum_t) * state[name]


        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if wandb_logger is not None:
            dict_info = loss_dict_reduced_scaled
            dict_info["lr"] = optimizer.param_groups[0]["lr"]
            if 'rel_batch' in loss_dict:
                dict_info['rel_batch'] = loss_dict['rel_batch'].item()
            wandb_logger.log(dict_info)
            

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break


    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


def drm_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, 
                    logger=None, ema_m=None, wandb_logger=None, 
                    model_t=None):

    # if get_rank() == 0:
    #     print("Trying to find SyncBN...")
    #     for m in model.modules():
    #         if isinstance(m, torch.nn.SyncBatchNorm):
    #             print("FOUND SyncBN:", m)

    # dist.barrier()
    # raise

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_update_interval = getattr(args, "teacher_update_interval", 1000)
    momentum_t = getattr(args, "teacher_momentum", 0.999)
    teacher_update = getattr(args, "teacher_update", False)


    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    def set_attr(model, name, value):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            setattr(model.module, name, value)
        else:
            setattr(model, name, value)

    set_attr(criterion, "ind_to_classes", data_loader.dataset.ind_to_classes)
    # criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    set_attr(criterion, "ind_to_predicates", data_loader.dataset.ind_to_predicates)
    set_attr(model, "ind_to_classes", data_loader.dataset.ind_to_classes)
    set_attr(model, "name2predicates", data_loader.dataset.name2predicates)
    set_attr(model, "name2classes", data_loader.dataset.name2classes)
    if model_t is not None:
        set_attr(model_t, "ind_to_classes", data_loader.dataset.ind_to_classes)
        set_attr(model_t, "name2predicates", data_loader.dataset.name2predicates)

    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in tqdm(metric_logger.log_every(data_loader, print_freq, header, logger=logger), total=len(data_loader), desc=f'train rank {get_rank()}', position=get_rank(), leave=True):
        args.global_iter += 1

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            try:
                if need_tgt_for_training:
                    outputs, _ = model(samples, targets, global_iter=args.global_iter)
                else:
                    outputs = model(samples, global_iter=args.global_iter)
            except:
                print("**************** failed forward *******************")
                print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
                traceback.print_exc()   # original error + full stack
                raise

            if model_t is not None:
                with torch.no_grad():
                    if need_tgt_for_training:
                        outputs_t, _ = model_t(samples, targets, global_iter=args.global_iter)
                    else:
                        outputs_t = model_t(samples, global_iter=args.global_iter)
            else:
                outputs_t = None 

            loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
                                global_iter=args.global_iter)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            for k, v in loss_dict.items():
                if v.grad_fn is None and k in weight_dict.keys():
                    print(f"[WARN] {k} has no grad_fn")
            try:
                losses.backward()
            except:
                print("**************** failed backward *******************")
                print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
                traceback.print_exc()   # original error + full stack
                raise

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if model_t is not None and teacher_update and args.global_iter % teacher_update_interval == 0:
            if utils.get_rank() == 0:
                print("*"*10, "global iter:", args.global_iter, 
                  " update teacher's weights with momentum:%s!" % momentum_t)
            for name, p in model_t.named_parameters():
                state = model.state_dict()
                keys = list(state.keys())
                if 'module.' in keys[0]:
                    name = 'module.' + name

                with torch.no_grad():
                    p.data = momentum_t * p.data + (1-momentum_t) * state[name]


        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if wandb_logger is not None:
            dict_info = loss_dict_reduced_scaled
            dict_info["lr"] = optimizer.param_groups[0]["lr"]
            if 'rel_batch' in loss_dict:
                dict_info['rel_batch'] = loss_dict['rel_batch'].item()
            wandb_logger.log(dict_info)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break


    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


def drm_train_one_epoch_pretrain(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, 
                    logger=None, ema_m=None, wandb_logger=None, 
                    model_t=None, postprocessors=None, data_loader_val=None, base_ds=None):

    # if get_rank() == 0:
    #     print("Trying to find SyncBN...")
    #     for m in model.modules():
    #         if isinstance(m, torch.nn.SyncBatchNorm):
    #             print("FOUND SyncBN:", m)

    # dist.barrier()
    # raise

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_update_interval = getattr(args, "teacher_update_interval", 1000)
    momentum_t = getattr(args, "teacher_momentum", 0.999)
    teacher_update = getattr(args, "teacher_update", False)


    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    def set_attr(model, name, value):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            setattr(model.module, name, value)
        else:
            setattr(model, name, value)

    set_attr(criterion, "ind_to_classes", data_loader.dataset.ind_to_classes)
    # criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    set_attr(criterion, "ind_to_predicates", data_loader.dataset.ind_to_predicates)
    set_attr(model, "ind_to_classes", data_loader.dataset.ind_to_classes)
    set_attr(model, "name2predicates", data_loader.dataset.name2predicates)
    set_attr(model, "name2classes", data_loader.dataset.name2classes)
    if model_t is not None:
        set_attr(model_t, "ind_to_classes", data_loader.dataset.ind_to_classes)
        set_attr(model_t, "name2predicates", data_loader.dataset.name2predicates)

    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    checkpoint_iter = 0

    for samples, targets in tqdm(metric_logger.log_every(data_loader, print_freq, header, logger=logger), total=len(data_loader), desc=f'train rank {get_rank()}', position=get_rank(), leave=True):
        model.train()
        criterion.train()
        args.global_iter += 1

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            try:
                if need_tgt_for_training:
                    outputs, _ = model(samples, targets, global_iter=args.global_iter)
                else:
                    outputs = model(samples, global_iter=args.global_iter)
            except:
                print("**************** failed forward *******************")
                print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
                traceback.print_exc()   # original error + full stack
                raise

            if model_t is not None:
                with torch.no_grad():
                    if need_tgt_for_training:
                        outputs_t, _ = model_t(samples, targets, global_iter=args.global_iter)
                    else:
                        outputs_t = model_t(samples, global_iter=args.global_iter)
            else:
                outputs_t = None 

            loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
                                global_iter=args.global_iter)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            for k, v in loss_dict.items():
                if v.grad_fn is None and k in weight_dict.keys():
                    print(f"[WARN] {k} has no grad_fn")
            try:
                losses.backward()
            except:
                print("**************** failed backward *******************")
                print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
                traceback.print_exc()   # original error + full stack
                raise

        # if get_rank() == 0:
        #     print_gpu_memory()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if model_t is not None and teacher_update and args.global_iter % teacher_update_interval == 0:
            if utils.get_rank() == 0:
                print("*"*10, "global iter:", args.global_iter, 
                  " update teacher's weights with momentum:%s!" % momentum_t)
            for name, p in model_t.named_parameters():
                state = model.state_dict()
                keys = list(state.keys())
                if 'module.' in keys[0]:
                    name = 'module.' + name

                with torch.no_grad():
                    p.data = momentum_t * p.data + (1-momentum_t) * state[name]


        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if wandb_logger is not None:
            dict_info = loss_dict_reduced_scaled
            dict_info["lr"] = optimizer.param_groups[0]["lr"]
            if 'rel_batch' in loss_dict:
                dict_info['rel_batch'] = loss_dict['rel_batch'].item()
            wandb_logger.log(dict_info)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

        checkpoint_iter += 1
        if checkpoint_iter % 5000 == 0:
            test_stats, coco_evaluator = drm_evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            r_at_50 = test_stats['R@50']

            def generate_model_name(epoch: int, r50: float) -> str:
                """
                Generate model name based on epoch, R@50 and current time.

                Args:
                    epoch (int): current epoch
                    r50 (float): Recall@50 value

                Returns:
                    str: model name
                """
                time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                return f"epoch{epoch}_R50_{r50:.2f}_{time_str}.pth"

            checkpoint_path = os.path.join(args.output_dir, generate_model_name(epoch, r_at_50))
            utils.save_on_master({
                'model': model.module.state_dict() if is_dist_avail_and_initialized() else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'R@50': r_at_50
            }, checkpoint_path)

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


def drm_train_one_epoch_pretrain_checkpoint_ver(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, 
                    logger=None, ema_m=None, wandb_logger=None, 
                    model_t=None, postprocessors=None, data_loader_val=None, base_ds=None, continue_iter=0):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_update_interval = getattr(args, "teacher_update_interval", 1000)
    momentum_t = getattr(args, "teacher_momentum", 0.999)
    teacher_update = getattr(args, "teacher_update", False)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    def set_attr(model, name, value):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            setattr(model.module, name, value)
        else:
            setattr(model, name, value)

    set_attr(criterion, "ind_to_classes", data_loader.dataset.ind_to_classes)
    # criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    set_attr(criterion, "ind_to_predicates", data_loader.dataset.ind_to_predicates)
    set_attr(model, "ind_to_classes", data_loader.dataset.ind_to_classes)
    set_attr(model, "name2predicates", data_loader.dataset.name2predicates)
    set_attr(model, "name2classes", data_loader.dataset.name2classes)
    if model_t is not None:
        set_attr(model_t, "ind_to_classes", data_loader.dataset.ind_to_classes)
        set_attr(model_t, "name2predicates", data_loader.dataset.name2predicates)
        set_attr(model_t, "name2classes", data_loader.dataset.name2classes)

    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    model.train()
    criterion.train()

    _cnt = 0
    current_iter = 0

    # -------------------- debug -------------------------
    # continue_iter = 10000000000
    # -------------------- debug -------------------------

    if continue_iter > 0:
        data_loader.dataset.resuming = True

    # set_timeout()

    # for samples, targets in tqdm(data_loader, total=len(data_loader), desc=f'train rank {get_rank()}', position=get_rank(), leave=True):
    # for samples, targets in tqdm(metric_logger.log_every(data_loader, print_freq, header, logger=logger), total=len(data_loader), desc=f'train rank {get_rank()}, epoch: {epoch}', position=get_rank(), leave=True):
    for samples, targets in tqdm(data_loader, total=len(data_loader), desc=f'train rank {get_rank()}, epoch: {epoch}', position=get_rank(), leave=True):
        # if get_rank() == 0:
        #     print("---------------------------------")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if get_rank() == 0:
            with open(os.path.join(args.output_dir, "heartbeat.txt"), "w") as f:
                f.write(str(time.time()))

        # time.sleep(3000)

        # print(f"Rank {get_rank()} index: {targets[0]['index']}")
        # print('get index!')
        if current_iter < continue_iter:
            current_iter += 1
            _cnt += 1
            if current_iter == (continue_iter - 1):
                data_loader.dataset.resuming = False
            continue

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        # set_timeout()

        args.global_iter += 1

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            try:
                if need_tgt_for_training:
                    outputs, _ = model(samples, targets, global_iter=args.global_iter)
                    # outputs = model(samples, targets, global_iter=args.global_iter)
                else:
                    outputs = model(samples, global_iter=args.global_iter)
                # print(f"Rank {get_rank()} done outputs")
            except:
                print("**************** failed forward *******************")
                print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
                traceback.print_exc()   # original error + full stack
                raise

            if model_t is not None:
                with torch.no_grad():
                    if need_tgt_for_training:
                        outputs_t, _ = model_t(samples, targets, global_iter=args.global_iter)
                        # outputs_t = model_t(samples, targets, global_iter=args.global_iter)
                    else:
                        outputs_t = model_t(samples, global_iter=args.global_iter)
            else:
                outputs_t = None

            # continue

            try:
                loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
                                    global_iter=args.global_iter)
                # print(f"Rank {get_rank()} done loss")
            except:
                print("**************** failed loss *******************")
                print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
                traceback.print_exc()   # original error + full stack
                raise

            # loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
            #                     global_iter=args.global_iter)

            weight_dict = criterion.weight_dict
            # print(loss_dict)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            print("**************** failed loss *******************")
            print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
            print(targets)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            for k, v in loss_dict.items():
                if v.grad_fn is None and k in weight_dict.keys():
                    print(f"[WARN] {k} has no grad_fn")
            try:
                losses.backward()
                # pass
            except:
                print("**************** failed backward *******************")
                print(f"Rank {get_rank()} | File name: {targets[0]['filename']} | Index: {targets[0]['index']}")
                traceback.print_exc()   # original error + full stack
                raise

        # print_gpu_memory()

        # if get_rank() == 0:
        #     print_gpu_memory()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if model_t is not None and teacher_update and args.global_iter % teacher_update_interval == 0:
            if utils.get_rank() == 0:
                print("*"*10, "global iter:", args.global_iter, 
                  " update teacher's weights with momentum:%s!" % momentum_t)
            for name, p in model_t.named_parameters():
                state = model.state_dict()
                keys = list(state.keys())
                if 'module.' in keys[0]:
                    name = 'module.' + name

                with torch.no_grad():
                    p.data = momentum_t * p.data + (1-momentum_t) * state[name]


        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if wandb_logger is not None:
            dict_info = loss_dict_reduced_scaled
            dict_info["lr"] = optimizer.param_groups[0]["lr"]
            if 'rel_batch' in loss_dict:
                dict_info['rel_batch'] = loss_dict['rel_batch'].item()
            wandb_logger.log(dict_info)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
        current_iter += 1

        if current_iter % 100 == 0 and current_iter != 0:
            utils.save_on_master({
                'model': model.module.state_dict() if is_dist_avail_and_initialized() else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'iter': current_iter  # Stores the iter needs to conduct now.
            }, os.path.join(args.output_dir, "tmp_checkpoint.pth"))

        # current_iter = 15000
        # if current_iter % 5000 == 0 and current_iter != 0:
        if current_iter % 5000 == 0 and current_iter != 0:
            def generate_model_name(epoch: int, r50: Optional[float] = None, current_iter: Optional[int] = None) -> str:
                """
                Generate model name based on epoch, R@50 and current time.

                Args:
                    epoch (int): current epoch
                    r50 (float): Recall@50 value

                Returns:
                    str: model name
                """
                time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                if current_iter is not None:
                    return f"epoch{epoch}_iter_{current_iter}_{time_str}.pth"
                elif r50 is not None:
                    return f"epoch{epoch}_R50_{r50:.2f}_{time_str}.pth"
                else:
                    return f"epoch{epoch}_pretest_{time_str}.pth"
            pre_test_checkpoint_path = os.path.join(args.output_dir, generate_model_name(epoch, current_iter=current_iter))
            utils.save_on_master({
                'model': model.module.state_dict() if is_dist_avail_and_initialized() else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'iter': current_iter  # Stores the iter needs to conduct now.
            }, pre_test_checkpoint_path)

            # test_stats, coco_evaluator = drm_evaluate(
            #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            #     wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            # )
            # r_at_50 = test_stats['R@50']

            # checkpoint_path = os.path.join(args.output_dir, generate_model_name(epoch, r_at_50))
            # utils.save_on_master({
            #     'model': model.module.state_dict() if is_dist_avail_and_initialized() else model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'lr_scheduler': lr_scheduler.state_dict(),
            #     'epoch': epoch,
            #     'args': args,
            #     'R@50': r_at_50,
            #     'iter': current_iter
            # }, checkpoint_path)

            # if os.path.exists(pre_test_checkpoint_path):
            #     os.remove(pre_test_checkpoint_path)
        if get_rank() == 0:
            print("---------------------------------")


    signal.alarm(0)  # cancel the timer

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


def train_one_epoch_resume(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
        lr_scheduler=None,
        args=None,
        model_t=None,
        continue_iter=0
        ):
    # print(f"Rank {get_rank()} reached start.")
    save_interval = 100

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    teacher_update_interval = getattr(args, "teacher_update_interval", 1000)
    momentum_t = getattr(args, "teacher_momentum", 0.999)
    teacher_update = getattr(args, "teacher_update", False)


    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 


    model.train()
    criterion.train()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0


    current_iter = 0
    if current_iter < continue_iter:
        data_loader.dataset.empty_output_flip = True
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
    # for samples, targets in tqdm(data_loader, total=len(data_loader), desc=f'Rank {get_rank()}:', position=get_rank()):
    for i, (samples, targets) in enumerate(data_loader):
        # print(f"Rank {get_rank()} reached here.")
        if current_iter < continue_iter:
            if current_iter == (continue_iter - 1):
                data_loader.dataset.empty_output_flip = False
            current_iter += 1
            continue

        # if current_iter == 2055:
        #     current_iter += 1
        #     continue

        args.global_iter += 1

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets, global_iter=args.global_iter)
            else:
                outputs = model(samples, global_iter=args.global_iter)
            
            if model_t is not None:
                with torch.no_grad():
                    if need_tgt_for_training:
                        outputs_t = model_t(samples, targets, global_iter=args.global_iter)
                    else:
                        outputs_t = model_t(samples, global_iter=args.global_iter)
            else:
                outputs_t = None 
                    
            loss_dict = criterion(outputs, targets, outputs_t=outputs_t, 
                                  global_iter=args.global_iter)

            weight_dict = criterion.weight_dict
            used_losses = {k: (loss_dict[k] * weight_dict[k]).item() for k in loss_dict.keys() if k in weight_dict}
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            # print(targets)
            print("Loss is {}, stopping training".format(loss_value))
            print(f"error iter: {current_iter}")
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            try:
                scaler.scale(losses).backward()
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is None:
                        print("NO GRAD:", name)
            except Exception as e:
                # print the detailed error
                print("An error occurred during backward pass:")
                print(traceback.format_exc())  # print the original traceback

                # print file_name
                # file_names = []
                # for target in targets:
                #     file_names.append(target['file_name'])
                # print(targets)
                print(f"error iter: {current_iter}")

                # re-raise
                raise e  # re-raise the caught exception
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if model_t is not None and teacher_update and args.global_iter % teacher_update_interval == 0:
            if utils.get_rank() == 0:
                print("*"*10, "global iter:", args.global_iter, 
                  " update teacher's weights with momentum:%s!" % momentum_t)
            for name, p in model_t.named_parameters():
                state = model.state_dict()
                keys = list(state.keys())
                if 'module.' in keys[0]:
                    name = 'module.' + name

                with torch.no_grad():
                    p.data = momentum_t * p.data + (1-momentum_t) * state[name]


        if args.onecyclelr:
            lr_scheduler.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # loss_print_dict = {
        #     "loss": loss_value,
        #     "bbox": loss_dict_reduced_scaled["loss_bbox"],
        #     "ce": loss_dict_reduced_scaled["loss_ce"],
        #     "edges": loss_dict_reduced_scaled["loss_edges"]
        # }

        if i % 100 == 0:
            if not is_distributed() or (is_distributed() and get_rank() == 0):

                # msg = f"Epoch {epoch} Iter {i}/{len(data_loader)}"

                # for k, v in loss_print_dict.items():
                #     msg += f" | {k}={v:.4f}"

                # print(msg)
                print(f"Epoch {epoch} Iter {i}/{len(data_loader)}")
                print(used_losses)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

        if is_distributed():
            dist.barrier()
        current_iter += 1
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'heartbeat.txt'), "w") as f:
                f.write(str(time.time()))
        if current_iter % save_interval == 0:
            if is_main_process():
                torch.save({
                    "model": model.module.state_dict() if is_distributed() else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "continue_iter": current_iter,
                }, os.path.join(args.output_dir, "checkpoint.pth"))


    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    # if getattr(criterion, 'loss_weight_decay', False):
    #     resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    # return resstat
    return None


@torch.no_grad()
def log_v2_distill_canaries(model, outputs, outputs_t, pool_spec, epoch, it):
    """Monitoring only -- has NO effect on training.

    1. GT-localization canary: per GT box, IoU between the student's box at
       ITS matched query and the teacher's box at ITS OWN matched query (each
       side re-anchors to GT via its own matcher). Both localize the same GT,
       so this should sit high (~0.85+); a slide below the threshold means the
       student's detector is drifting away from the teacher -- distillation
       targets are then computed on diverging upstreams.
    2. Attn-gate readout: the rln decoder self-attn gate scalars, printed raw.
       The pretrain checkpoint already moved them away from their init, so no
       absolute threshold is meaningful -- watch the trend (a sustained slide
       toward 0 would mean the student mutes context to cheat the distill loss).
    """
    def cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    ious = []
    boxes_s_all = outputs['pred_boxes']
    boxes_t_all = outputs_t['pred_boxes']
    pool_spec_t = outputs_t.get('pool_spec', None) if isinstance(outputs_t, dict) else None
    for i, spec in enumerate(pool_spec):
        q_s = spec['gt_q']
        q_t = pool_spec_t[i]['gt_q'] if pool_spec_t is not None else q_s
        if q_s.numel() == 0:
            continue
        bs_ = cxcywh_to_xyxy(boxes_s_all[i, q_s].float())
        bt_ = cxcywh_to_xyxy(boxes_t_all[i, q_t].float())
        lt = torch.max(bs_[:, :2], bt_[:, :2])
        rb = torch.min(bs_[:, 2:], bt_[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area_s = (bs_[:, 2] - bs_[:, 0]).clamp(min=0) * (bs_[:, 3] - bs_[:, 1]).clamp(min=0)
        area_t = (bt_[:, 2] - bt_[:, 0]).clamp(min=0) * (bt_[:, 3] - bt_[:, 1]).clamp(min=0)
        ious.append(inter / (area_s + area_t - inter + 1e-6))

    msg = f"[distill-canary] epoch {epoch} iter {it}:"
    if ious:
        mean_iou = torch.cat(ious).mean().item()
        msg += f" gt-slot student/teacher box IoU = {mean_iou:.4f}"
        if mean_iou < 0.7:
            msg += "  << WARNING: student/teacher disagree on GT locations (IoU < 0.7) -- upstream drifting apart, distill targets unreliable >>"
    else:
        msg += " gt-slot IoU: n/a (no GT slots in this batch)"

    model_no_ddp = model.module if hasattr(model, 'module') else model
    gates = [(n, p.detach().item()) for n, p in model_no_ddp.named_parameters() if 'self_attn_gate' in n]
    if gates:
        msg += " | attn gates = [" + ", ".join(f"{v:.3f}" for _, v in gates) + "]"
    print(msg)


def train_one_epoch_new(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
        lr_scheduler=None,
        args=None,
        model_t=None,
        continue_iter=0
        ):
    criterion.ind_to_classes = data_loader.dataset.ind_to_classes
    criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates

    if is_distributed():
        model.module.name2classes = data_loader.dataset.name2classes
    else:
        model.name2classes = data_loader.dataset.name2classes

    save_interval = 100
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # model.to(f'cuda:{get_rank()}')
    # criterion.to(f'cuda:{get_rank()}')

    model.train()
    criterion.train()

    total_iter = len(data_loader)
    current_iter = 0
    if current_iter < continue_iter:
        data_loader.dataset.empty_output_flip = True
    for i, (samples, targets) in enumerate(data_loader):
        if current_iter < continue_iter:
            if current_iter == (continue_iter - 1):
                data_loader.dataset.empty_output_flip = False
            current_iter += 1
            continue

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        for target in targets:
            target['current_iter'] = current_iter
            target['total_iter'] = total_iter

        if args.modelname in ['ovsgtr_rln_decoder_pair_pruning_v6', 'ovsgtr_rln_decoder_pair_pruning_v7', 'ovsgtr_base']:
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs, permute_idx = model(samples, targets)

                if model_t is not None:
                    with torch.no_grad():
                        # if is_main_process():
                        #     print('teacher:')
                        #     print(targets[0])
                        outputs_t, permute_idx = model_t(samples, targets, permute_idx)
                else:
                    outputs_t = None

                loss_dict = criterion(outputs, targets, outputs_t=outputs_t)

                weight_dict = criterion.weight_dict
                # used_losses = {k: (loss_dict[k] * weight_dict[k]).item() for k in loss_dict.keys() if k in weight_dict}
                used_losses = {k: f"{(loss_dict[k] * weight_dict[k]).item():.4f}" for k in loss_dict.keys() if k in weight_dict}
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        else:
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(samples, targets)

                if model_t is not None:
                    with torch.no_grad():
                        # V2 gt_query distillation: feed the student's entity-pool
                        # sampling state to the frozen teacher so both build the
                        # same pool (V6's permute_idx analog). Other models don't
                        # emit pool_spec and keep the old call.
                        _pool_spec = outputs.get('pool_spec', None) if isinstance(outputs, dict) else None
                        if _pool_spec is not None:
                            outputs_t = model_t(samples, targets, pool_spec=_pool_spec)
                            # Monitoring only (query-correspondence IoU + attn gates).
                            if i % 100 == 0 and utils.is_main_process():
                                log_v2_distill_canaries(model, outputs, outputs_t, _pool_spec, epoch, i)
                        else:
                            outputs_t = model_t(samples, targets)
                else:
                    outputs_t = None

                loss_dict = criterion(outputs, targets, outputs_t=outputs_t)

                weight_dict = criterion.weight_dict
                # used_losses = {k: (loss_dict[k] * weight_dict[k]).item() for k in loss_dict.keys() if k in weight_dict}
                used_losses = {k: f"{(loss_dict[k] * weight_dict[k]).item():.4f}" for k in loss_dict.keys() if k in weight_dict}
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if torch.isinf(losses).any():
            print("Loss contains inf, stop training.")
            print(f"error iter: {current_iter}")
            print(loss_dict)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is None:
                    print("NO GRAD:", name)

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if i % 100 == 0:
            if not is_distributed() or (is_distributed() and get_rank() == 0):
                print(f"Epoch {epoch} Iter {i}/{len(data_loader)}")
                print(used_losses)

        if is_distributed():
            dist.barrier()
        current_iter += 1
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'heartbeat.txt'), "w") as f:
                f.write(str(time.time()))
        if current_iter % save_interval == 0:
            if is_main_process():
                torch.save({
                    "model": model.module.state_dict() if is_distributed() else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "continue_iter": current_iter,
                }, os.path.join(args.output_dir, "checkpoint.pth"))

                torch.save({
                    "model": model.module.state_dict() if is_distributed() else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "continue_iter": current_iter,
                }, os.path.join(args.output_dir, "checkpoint_backup.pth"))

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    return None


# def train_one_epoch_new(
#         model: torch.nn.Module,
#         criterion: torch.nn.Module,
#         data_loader: Iterable,
#         optimizer: torch.optim.Optimizer,
#         device: torch.device,
#         epoch: int,
#         max_norm: float = 0,
#         lr_scheduler=None,
#         args=None,
#         model_t=None,
#         continue_iter=0
#         ):
#     criterion.ind_to_classes = data_loader.dataset.ind_to_classes
#     criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates

#     if is_distributed():
#         model.module.name2classes = data_loader.dataset.name2classes
#     else:
#         model.name2classes = data_loader.dataset.name2classes

#     save_interval = 100
#     scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

#     # model.to(f'cuda:{get_rank()}')
#     # criterion.to(f'cuda:{get_rank()}')

#     model.train()
#     criterion.train()

#     current_iter = 0
#     if current_iter < continue_iter:
#         data_loader.dataset.empty_output_flip = True
#     for i, (samples, targets) in enumerate(data_loader):
#         if current_iter < continue_iter:
#             if current_iter == (continue_iter - 1):
#                 data_loader.dataset.empty_output_flip = False
#             current_iter += 1
#             continue

#         samples = samples.to(device)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

#         with torch.cuda.amp.autocast(enabled=args.amp):
#             outputs = model(samples, targets)

#             if model_t is not None:
#                 with torch.no_grad():
#                     outputs_t = model_t(samples, targets)
#             else:
#                 outputs_t = None
                    
#             loss_dict = criterion(outputs, targets, outputs_t=outputs_t)

#             weight_dict = criterion.weight_dict
#             # used_losses = {k: (loss_dict[k] * weight_dict[k]).item() for k in loss_dict.keys() if k in weight_dict}
#             used_losses = {k: f"{(loss_dict[k] * weight_dict[k]).item():.4f}" for k in loss_dict.keys() if k in weight_dict}
#             losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#         if torch.isinf(losses).any():
#             print("Loss contains inf, stop training.")
#             print(f"error iter: {current_iter}")
#             print(loss_dict)
#             sys.exit(1)

#         # amp backward function
#         if args.amp:
#             optimizer.zero_grad()
#             scaler.scale(losses).backward()
#             for name, p in model.named_parameters():
#                 if p.requires_grad and p.grad is None:
#                     print("NO GRAD:", name)

#             if max_norm > 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             # original backward function
#             optimizer.zero_grad()
#             losses.backward()
#             if max_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()

#         if i % 100 == 0:
#             if not is_distributed() or (is_distributed() and get_rank() == 0):
#                 print(f"Epoch {epoch} Iter {i}/{len(data_loader)}")
#                 print(used_losses)

#         if is_distributed():
#             dist.barrier()
#         current_iter += 1
#         if utils.is_main_process():
#             with open(os.path.join(args.output_dir, 'heartbeat.txt'), "w") as f:
#                 f.write(str(time.time()))
#         if current_iter % save_interval == 0:
#             if is_main_process():
#                 torch.save({
#                     "model": model.module.state_dict() if is_distributed() else model.state_dict(),
#                     "optimizer": optimizer.state_dict(),
#                     "lr_scheduler": lr_scheduler.state_dict(),
#                     "epoch": epoch,
#                     "args": args,
#                     "continue_iter": current_iter,
#                 }, os.path.join(args.output_dir, "checkpoint.pth"))

#                 torch.save({
#                     "model": model.module.state_dict() if is_distributed() else model.state_dict(),
#                     "optimizer": optimizer.state_dict(),
#                     "lr_scheduler": lr_scheduler.state_dict(),
#                     "epoch": epoch,
#                     "args": args,
#                     "continue_iter": current_iter,
#                 }, os.path.join(args.output_dir, "checkpoint_backup.pth"))

#     if getattr(criterion, 'loss_weight_decay', False):
#         criterion.loss_weight_decay(epoch=epoch)
#     if getattr(criterion, 'tuning_matching', False):
#         criterion.tuning_matching(epoch)

#     return None


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    postprocessors['bbox'].eval()
    try:
        criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    except:
        pass

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    try:
        assert args.dataset_file == "coco"
        #coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
        coco_evaluator = None 
    except:
        coco_evaluator = None 

    sgg_evaluator = None 
    # "gqa" is GQA200 wearing VG's clothes (see datasets/vg.py:build_gqa);
    # it must reach the SGG evaluator, or eval finishes silently with no R@K.
    if args.dataset_file in ["vg", "gqa", "oicap", "coco", "coco_flickr30k", "coco_flickr30k_sbucaptions"]:
        iou_types = list(iou_types)
        if args is not None and getattr(args, "iou_types", None):
            iou_types = list(args.iou_types)
        else:
            iou_types.append("relation")

        iou_types = tuple(set(iou_types))
        try:
            num_rel_category = args.num_rel_category
        except:
            num_rel_category = 51

        sgg_evaluator = SggEvaluator(data_loader.dataset, iou_types,  # ['bbox', 'relation']
                                     mode="sgdet", num_rel_category=num_rel_category,  # 51
                                     multiple_preds=False, iou_thres=0.5,
                                     output_folder=os.path.join(output_dir, "sgg_eval"),
                                     ovd_enabled=getattr(args, "sg_ovd_mode", False),
                                     ovr_enabled=getattr(args, "sg_ovr_mode", False)
                                     )
        postprocessors['bbox'].eval()


    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 

        postprocessors['bbox'].name2classes = data_loader.dataset.name2classes
        do_sgg = getattr(postprocessors['bbox'], 'do_sgg', False)
        if do_sgg:
            postprocessors['bbox'].name2predicates = data_loader.dataset.name2predicates



    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}


        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        if postprocessors['bbox'].use_gt_box:
            match_ids = postprocessors['bbox'].matcher(outputs, targets)
            tgt_dicts = [{'ids': ids, 'gt_boxes': target['boxes'],
                          'gt_labels': target['labels']}
                          for target, ids in zip(targets, match_ids)]

            results = postprocessors['bbox'](outputs, orig_target_sizes, 
                                             gt_dicts=tgt_dicts)
        else:
            results = postprocessors['bbox'](outputs, orig_target_sizes)

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id']: output for target, output in zip(targets, results)}

        #torch.save(res, os.path.join(output_dir, "%s.pt"%_cnt ))
        #res = torch.load(os.path.join(output_dir, "%s.pt"%_cnt))

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if sgg_evaluator is not None:
            sgg_evaluator.update(res)

        _cnt += 1
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"]
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if sgg_evaluator is not None:
        sgg_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    sgg_res = None
    if sgg_evaluator is not None:
        sgg_res = sgg_evaluator.accumulate()
        sgg_evaluator.summarize()
        sgg_evaluator.reset()

        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    if sgg_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            try:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats.tolist()
            except:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats

        if sgg_res is not None:
            stats.update(sgg_res)



    return stats, coco_evaluator


@torch.no_grad()
def drm_evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    postprocessors['bbox'].eval()

    def set_attr(model, name, value):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            setattr(model.module, name, value)
        else:
            setattr(model, name, value)

    set_attr(criterion, "ind_to_classes", data_loader.dataset.ind_to_classes)
    # criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    set_attr(criterion, "ind_to_predicates", data_loader.dataset.ind_to_predicates)
    set_attr(model, "ind_to_classes", data_loader.dataset.ind_to_classes)
    set_attr(model, "name2predicates", data_loader.dataset.name2predicates)
    set_attr(model, "name2classes", data_loader.dataset.name2classes)
    # if model_t is not None:
    #     set_attr(model_t, "ind_to_classes", data_loader.dataset.ind_to_classes)
    #     set_attr(model_t, "name2predicates", data_loader.dataset.name2predicates)
    #     set_attr(model_t, "name2classes", data_loader.dataset.name2classes)

    # try:
    #     criterion.ind_to_classes = data_loader.dataset.ind_to_classes
    #     criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    #     postprocessors['bbox'].ind_to_classes = data_loader.dataset.ind_to_classes
    # except:
    #     pass

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    try:
        assert args.dataset_file == "coco"
        #coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
        coco_evaluator = None 
    except:
        coco_evaluator = None 

    sgg_evaluator = None 
    # "gqa" is GQA200 wearing VG's clothes (see datasets/vg.py:build_gqa);
    # it must reach the SGG evaluator, or eval finishes silently with no R@K.
    if args.dataset_file in ["vg", "gqa", "vg_contrastive", "oicap", "coco", "coco_pretrain", "coco_flickr30k", "coco_flickr30k_sbucaptions"]:
        iou_types = list(iou_types)
        if args is not None and getattr(args, "iou_types", None):
            iou_types = list(args.iou_types)
        else:
            iou_types.append("relation")

        iou_types = tuple(set(iou_types))
        try:
            num_rel_category = args.num_rel_category
        except:
            num_rel_category = 51

        sgg_evaluator = SggEvaluator(data_loader.dataset, iou_types,
                                     mode="sgdet", num_rel_category=num_rel_category,
                                     multiple_preds=False, iou_thres=0.5,
                                     output_folder=os.path.join(output_dir, "sgg_eval"),
                                     ovd_enabled=getattr(args, "sg_ovd_mode", False),
                                     ovr_enabled=getattr(args, "sg_ovr_mode", False)
                                     )
        postprocessors['bbox'].eval()


    use_text_labels = getattr(args, "use_text_labels", False)
    if use_text_labels:
        need_tgt_for_training = True 

        postprocessors['bbox'].name2classes = data_loader.dataset.name2classes
        do_sgg = getattr(postprocessors['bbox'], 'do_sgg', False)
        if do_sgg:
            postprocessors['bbox'].name2predicates = data_loader.dataset.name2predicates



    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    # model.name2predicates = data_loader.dataset.name2predicates
    # model.ind_to_classes = data_loader.dataset.ind_to_classes

    for samples, targets in tqdm(metric_logger.log_every(data_loader, 10, header, logger=logger), total=len(data_loader), desc=f'eval rank {get_rank()}', position=get_rank(), leave=True):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, results = model(samples, targets)
            else:
                outputs = model(samples)

            # dist.barrier()

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}


        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        # if postprocessors['bbox'].use_gt_box:
        #     match_ids = postprocessors['bbox'].matcher(outputs, targets)
        #     tgt_dicts = [{'ids': ids, 'gt_boxes': target['boxes'],
        #                   'gt_labels': target['labels']}
        #                   for target, ids in zip(targets, match_ids)]

        #     results = postprocessors['bbox'](outputs, orig_target_sizes, 
        #                                      gt_dicts=tgt_dicts)
        # else:
        #     if args.dataset_file == "coco_pretrain":
        #         cat_list = targets[0]['rel_caption'].split(".")
        #         cat_list = [x.strip().lower() for x in cat_list if x.strip()]
        #         results = postprocessors['bbox'](outputs, orig_target_sizes, cat_list=cat_list)
        #     else:
        #         results = postprocessors['bbox'](outputs, orig_target_sizes)

        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id']: output for target, output in zip(targets, results)}

        #torch.save(res, os.path.join(output_dir, "%s.pt"%_cnt ))
        #res = torch.load(os.path.join(output_dir, "%s.pt"%_cnt))

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if sgg_evaluator is not None:
            sgg_evaluator.update(res)

        _cnt += 1
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"]
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if sgg_evaluator is not None:
        sgg_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    sgg_res = None
    if sgg_evaluator is not None:
        sgg_res = sgg_evaluator.accumulate()
        if args.dataset_file != "coco_pretrain":
            sgg_evaluator.summarize()
        sgg_evaluator.reset()


    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    if sgg_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            try:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats.tolist()
            except:
                stats['coco_eval_bbox'] = sgg_evaluator.coco_eval['bbox'].stats

        if sgg_res is not None:
            stats.update(sgg_res)



    return stats, coco_evaluator


# @torch.no_grad()
# def evaluate_new(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
#     criterion.ind_to_classes = data_loader.dataset.ind_to_classes
#     criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
#     postprocessors['bbox'].ind_to_classes = data_loader.dataset.ind_to_classes
#     postprocessors['bbox'].ind_to_predicates = data_loader.dataset.ind_to_predicates
#     postprocessors['bbox'].name2predicates = data_loader.dataset.name2predicates
#     postprocessors['bbox'].name2classes = data_loader.dataset.name2classes
#     if is_dist_avail_and_initialized():
#         model.module.name2predicates = data_loader.dataset.name2predicates
#         model.module.name2classes = data_loader.dataset.name2classes
#         if hasattr(model.module, "monitor"):
#             model.module.monitor.name2predicates = data_loader.dataset.name2predicates
#             model.module.monitor.name2classes = data_loader.dataset.name2classes
#     else:
#         model.name2predicates = data_loader.dataset.name2predicates
#         model.name2classes = data_loader.dataset.name2classes
#         if hasattr(model, "monitor"):
#             model.monitor.name2predicates = data_loader.dataset.name2predicates
#             model.monitor.name2classes = data_loader.dataset.name2classes

#     model.eval()
#     criterion.eval()
#     postprocessors['bbox'].eval()

#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

#     sgg_evaluator = None 
#     if args.dataset_file in ["vg", "vg_contrastive", "oicap", "coco", "coco_pretrain", "coco_flickr30k", "coco_flickr30k_sbucaptions"]:
#         iou_types = list(iou_types)
#         iou_types.append("relation")
#         iou_types = tuple(set(iou_types))

#         # num_rel_category = args.num_rel_category
#         num_rel_category = 51  # TODO what does this do?

#         sgg_evaluator = SggEvaluator(
#             data_loader.dataset,
#             iou_types,
#             mode="sgdet",
#             num_rel_category=num_rel_category,
#             multiple_preds=False,
#             iou_thres=0.5,
#             output_folder=os.path.join(output_dir, "sgg_eval"),
#             ovd_enabled=getattr(args, "sg_ovd_mode", False),
#             ovr_enabled=getattr(args, "sg_ovr_mode", False)
#             )

#     weight_dict = criterion.weight_dict
#     average_loss_dict = {k: torch.tensor(0.0).to(device) for k in weight_dict}
#     total_iter = 0

#     for i, (samples, targets) in enumerate(data_loader):
#         total_iter += 1
#         samples = samples.to(device)
#         targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

#         with torch.cuda.amp.autocast(enabled=args.amp):
#             outputs = model(samples, targets)
#             loss_dict = criterion(outputs, targets)

#             # used_losses = {k: (loss_dict[k] * weight_dict[k]).item() for k in loss_dict.keys() if k in weight_dict}
#             used_losses = {k: loss_dict[k] for k in loss_dict.keys() if k in weight_dict}
#             for k, v in used_losses.items():
#                 average_loss_dict[k] += v

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         with torch.cuda.amp.autocast(enabled=args.amp):
#             if postprocessors['bbox'].use_gt_box:
#                 match_ids = postprocessors['bbox'].matcher(outputs, targets)
#                 tgt_dicts = [{'ids': ids, 'gt_boxes': target['boxes'], 'gt_labels': target['labels']} for target, ids in zip(targets, match_ids)]
#                 results = postprocessors['bbox'](outputs, orig_target_sizes, gt_dicts=tgt_dicts)
#             else:
#                 # cat_list = targets[0]['rel_caption'].split(".")
#                 # cat_list = [x.strip().lower() for x in cat_list if x.strip()]
#                 # results = postprocessors['bbox'](outputs, orig_target_sizes, cat_list=cat_list)
#                 results = postprocessors['bbox'](outputs, orig_target_sizes)  # TODO: make this compatible to OV OBD

#         res = {target['image_id']: output for target, output in zip(targets, results)}

#         if sgg_evaluator is not None:
#             sgg_evaluator.update(res)

#         if i % 100 == 0:
#             if not is_distributed() or (is_distributed() and get_rank() == 0):
#                 # print(f"Iter {i}/{len(data_loader)}")
#                 # print(used_losses)
#                 used_losses = {k: v.item() for k, v in used_losses.items()}
#                 print(f"Iter {i}/{len(data_loader)} | losses: {used_losses}", flush=True)

#     def reduce_loss_dict(loss_dict):
#         with torch.no_grad():
#             keys = sorted(loss_dict.keys())  # keep the order consistent
#             losses = torch.stack([loss_dict[k] for k in keys])  # (N,)

#             dist.all_reduce(losses, op=dist.ReduceOp.SUM)

#             reduced_dict = {k: v for k, v in zip(keys, losses)}
#         return reduced_dict

#     if dist.is_available() and dist.is_initialized():
#         world_size = dist.get_world_size()
#     else:
#         world_size = 1

#     if is_dist_avail_and_initialized():
#         average_loss_dict = reduce_loss_dict(average_loss_dict)
#     average_loss_dict = {k: (v / (world_size * total_iter)).item() for k, v in average_loss_dict.items()}

#     print(f"average loss: {average_loss_dict}")

#     if sgg_evaluator is not None:
#         sgg_evaluator.synchronize_between_processes()

#     if sgg_evaluator is not None:
#         _ = sgg_evaluator.accumulate()  # This step generates all sgg metrics.
#         if args.dataset_file != "coco_pretrain":
#             sgg_evaluator.summarize()  # This step generates all OBD metrics.
#         sgg_evaluator.reset()

#     # ==================== monitor part ====================
#     if hasattr(model.module if is_distributed() else model, "monitor"):
#         monitor_summarize = model.module.monitor.summarize() if is_distributed() else model.monitor.summarize()
#         if is_distributed():
#             if get_rank() == 0:
#                 print(monitor_summarize)
#         else:
#             print(monitor_summarize)

#     return None


@torch.no_grad()
def evaluate_new(model, criterion, postprocessors, data_loader, device, output_dir, args=None):
    criterion.ind_to_classes = data_loader.dataset.ind_to_classes
    criterion.ind_to_predicates = data_loader.dataset.ind_to_predicates
    postprocessors['bbox'].ind_to_classes = data_loader.dataset.ind_to_classes
    postprocessors['bbox'].ind_to_predicates = data_loader.dataset.ind_to_predicates
    postprocessors['bbox'].name2predicates = data_loader.dataset.name2predicates
    postprocessors['bbox'].name2classes = data_loader.dataset.name2classes
    if is_dist_avail_and_initialized():
        model.module.name2predicates = data_loader.dataset.name2predicates
        model.module.name2classes = data_loader.dataset.name2classes
        if hasattr(model.module, "monitor"):
            model.module.monitor.name2predicates = data_loader.dataset.name2predicates
            model.module.monitor.name2classes = data_loader.dataset.name2classes
    else:
        model.name2predicates = data_loader.dataset.name2predicates
        model.name2classes = data_loader.dataset.name2classes
        if hasattr(model, "monitor"):
            model.monitor.name2predicates = data_loader.dataset.name2predicates
            model.monitor.name2classes = data_loader.dataset.name2classes

    model.eval()
    criterion.eval()
    postprocessors['bbox'].eval()

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    sgg_evaluator = None 
    # "gqa" is GQA200 wearing VG's clothes (see datasets/vg.py:build_gqa);
    # it must reach the SGG evaluator, or eval finishes silently with no R@K.
    if args.dataset_file in ["vg", "gqa", "vg_contrastive", "oicap", "coco", "coco_pretrain", "coco_flickr30k", "coco_flickr30k_sbucaptions"]:
        iou_types = list(iou_types)
        iou_types.append("relation")
        iou_types = tuple(set(iou_types))

        # Sizes the mean-recall tables. Their category *names* already come
        # from dataset.ind_to_predicates (see SggEvaluator.get_sgg_evaluator),
        # so taking the count from anywhere else lets the two disagree: with a
        # hard-coded 51 and GQA200's 101 predicates, collect_mean_recall_items
        # indexes recall_count out of range. Same source, no disagreement.
        # 51 for VG150, so this is an identity there.
        num_rel_category = len(data_loader.dataset.ind_to_predicates)

        sgg_evaluator = SggEvaluator(
            data_loader.dataset,
            iou_types,
            mode="sgdet",
            num_rel_category=num_rel_category,
            multiple_preds=False,
            iou_thres=0.5,
            output_folder=os.path.join(output_dir),
            ovd_enabled=getattr(args, "sg_ovd_mode", False),
            ovr_enabled=getattr(args, "sg_ovr_mode", False)
            )

    weight_dict = criterion.weight_dict
    average_loss_dict = {k: torch.tensor(0.0).to(device) for k in weight_dict}
    total_iter = 0

    current_iter = 0
    whole_length_of_dataloader = len(data_loader)
    for i, (samples, targets) in enumerate(data_loader):
        total_iter += 1
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        for target in targets:
            target['current_iter'] = current_iter
            target['total_iter'] = whole_length_of_dataloader

        with torch.cuda.amp.autocast(enabled=args.amp):

            if args.modelname in ['ovsgtr_rln_decoder_pair_pruning_v6', 'ovsgtr_rln_decoder_pair_pruning_v7', 'ovsgtr_base']:
                outputs, permute_idx = model(samples, targets)
            else:
                outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            # used_losses = {k: (loss_dict[k] * weight_dict[k]).item() for k in loss_dict.keys() if k in weight_dict}
            used_losses = {k: loss_dict[k] for k in loss_dict.keys() if k in weight_dict}
            for k, v in used_losses.items():
                average_loss_dict[k] += v

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if postprocessors['bbox'].use_gt_box:
                match_ids = postprocessors['bbox'].matcher(outputs, targets)
                tgt_dicts = [{'ids': ids, 'gt_boxes': target['boxes'], 'gt_labels': target['labels']} for target, ids in zip(targets, match_ids)]
                # results = postprocessors['bbox'](outputs, orig_target_sizes, gt_dicts=tgt_dicts)
                results = postprocessors['bbox'](outputs, orig_target_sizes, targets, gt_dicts=tgt_dicts)
            else:
                # cat_list = targets[0]['rel_caption'].split(".")
                # cat_list = [x.strip().lower() for x in cat_list if x.strip()]
                # results = postprocessors['bbox'](outputs, orig_target_sizes, cat_list=cat_list)
                # results = postprocessors['bbox'](outputs, orig_target_sizes)
                results = postprocessors['bbox'](outputs, orig_target_sizes, targets)  # TODO: make this compatible to OV OBD

        res = {target['image_id']: output for target, output in zip(targets, results)}

        if sgg_evaluator is not None:
            sgg_evaluator.update(res)

        if i % 100 == 0:
            if not is_distributed() or (is_distributed() and get_rank() == 0):
                # print(f"Iter {i}/{len(data_loader)}")
                # print(used_losses)
                used_losses = {k: v.item() for k, v in used_losses.items()}
                print(f"Iter {i}/{len(data_loader)} | losses: {used_losses}", flush=True)
                with open(os.path.join(output_dir, 'output.txt'), "w", encoding="utf-8") as f:
                    f.write(f"Iter {i}/{len(data_loader)} | losses: {used_losses}")

        current_iter += 1

    def reduce_loss_dict(loss_dict):
        with torch.no_grad():
            keys = sorted(loss_dict.keys())  # keep the order consistent
            losses = torch.stack([loss_dict[k] for k in keys])  # (N,)

            dist.all_reduce(losses, op=dist.ReduceOp.SUM)

            reduced_dict = {k: v for k, v in zip(keys, losses)}
        return reduced_dict

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if is_dist_avail_and_initialized():
        average_loss_dict = reduce_loss_dict(average_loss_dict)
    average_loss_dict = {k: (v / (world_size * total_iter)).item() for k, v in average_loss_dict.items()}

    print(f"average loss: {average_loss_dict}")

    if sgg_evaluator is not None:
        sgg_evaluator.synchronize_between_processes()

    if sgg_evaluator is not None:
        _ = sgg_evaluator.accumulate()  # This step generates all sgg metrics.
        if args.dataset_file != "coco_pretrain":
            sgg_evaluator.summarize()  # This step generates all OBD metrics.
        sgg_evaluator.reset()

    # ==================== monitor part ====================
    if hasattr(model.module if is_distributed() else model, "monitor"):
        monitor_summarize = model.module.monitor.summarize() if is_distributed() else model.monitor.summarize()
        if is_distributed():
            if get_rank() == 0:
                print(monitor_summarize)
        else:
            print(monitor_summarize)

    return None


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id']: output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": image_id, 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res


