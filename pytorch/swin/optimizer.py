# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# This file has been modified by Graphcore Ltd.
import torch
from torch import optim as optim
from poptorch.optim import AdamW, SGD


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = SGD(parameters,
                        lr=config.TRAIN.BASE_LR,
                        momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                        weight_decay=config.TRAIN.WEIGHT_DECAY,
                        loss_scaling=config.TRAIN.LOSS_SCALING,
                        accum_type=torch.float16,
                        use_combined_accum=True)

    elif opt_lower == 'adamw':
        optimizer = AdamW(parameters,
                          lr=config.TRAIN.BASE_LR,
                          betas=config.TRAIN.OPTIMIZER.BETAS,
                          eps=config.TRAIN.OPTIMIZER.EPS,
                          weight_decay=config.TRAIN.WEIGHT_DECAY,
                          loss_scaling=config.TRAIN.LOSS_SCALING,
                          accum_type=torch.float32,
                          first_order_momentum_accum_type=torch.float32,
                          second_order_momentum_accum_type=torch.float32)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(
                param.shape) == 1 or name.endswith(".bias") or (
                name in skip_list) or check_keywords_in_name(
                name,
                skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
