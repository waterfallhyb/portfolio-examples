# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import get_constant_schedule


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_lr_scheduler(optimizer,
                     scheduler_type,
                     warmup_steps=None,
                     num_steps=None,
                     last_epoch=-1):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_steps)
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_steps, last_epoch=last_epoch)
    else:
        raise ValueError("Unknown scheduler_type:", scheduler_type)
    return scheduler


class NameScope:
    """ Create a name scope for a code block. All operators originating
        from this block will have their names prefixed by the given string.

        >>> with NameScope("CustomString"):
        ...     y = self.bmm(a, b)
        ...     z = torch.relu(y)
    """

    def __init__(self, name: str):
        assert isinstance(name, str), 'Parameter to NameScope must be a string'
        self.name = name

    def __enter__(self):
        torch.ops.poptorch.push_name_scope(self.name)

    def __exit__(self, type, value, traceback):
        torch.ops.poptorch.pop_name_scope()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}\tval:{val' + self.fmt + \
            '}\tavg:({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
