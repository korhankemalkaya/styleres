# python3.7
"""Misc utility functions."""

import os
import sys
import subprocess
from importlib import import_module
import argparse
from easydict import EasyDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import math
__all__ = [
    'init_dist', 'bool_parser', 'DictAction', 'parse_config', 'update_config', 'assert_config'
]


def init_dist(launcher, backend='nccl', **kwargs):
    """Initializes distributed environment."""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        port = os.environ.get('PORT', 29500)
        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        dist.init_process_group(backend=backend)
    else:
        raise NotImplementedError(f'Not implemented launcher type: '
                                  f'`{launcher}`!')

def bool_parser(arg):
    """Parses an argument to boolean."""
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ['1', 'true', 't', 'yes', 'y']:
        return True
    if arg.lower() in ['0', 'false', 'f', 'no', 'n']:
        return False
    raise argparse.ArgumentTypeError(f'`{arg}` cannot be converted to boolean!')


class DictAction(argparse.Action):
    """Argparse action to split an argument into key-value.

    NOTE: This class is borrowed from
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return val.lower() == 'true'
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def parse_config(config_file):
    """Parses configuration from python file."""
    assert os.path.isfile(config_file)
    directory = os.path.dirname(config_file)
    filename = os.path.basename(config_file)
    module_name, extension = os.path.splitext(filename)
    assert extension == '.py'
    sys.path.insert(0, directory)
    module = import_module(module_name)
    sys.path.pop(0)
    config = EasyDict()
    for key, value in module.__dict__.items():
        if key.startswith('__'):
            continue
        config[key] = value
    del sys.modules[module_name]
    return config


def update_config(config, new_config):
    """Updates configuration in a hierarchical level.

    For key-value pair {'a.b.c.d': v} in `new_config`, the `config` will be
    updated by

    config['a']['b']['c']['d'] = v
    """
    if new_config is None:
        return config

    assert isinstance(config, dict)
    assert isinstance(new_config, dict)

    for key, val in new_config.items():
        hierarchical_keys = key.split('.')
        temp = config
        for sub_key in hierarchical_keys[:-1]:
            temp = temp[sub_key]
        temp[hierarchical_keys[-1]] = val

    return config

def assert_config(config):
    num_gpus = config.num_gpus
    batch_size = config.batch_size
    val_batch_size = config.val_batch_size
    fid_num = config.controllers.FIDEvaluator.num
    use_ratio_loss = config.loss.e_loss_kwargs.ratio_lw > 0
    mixing_method = config.mixing_method

    #assert math.log(batch_size, 2).is_integer(), "Batch size should be power of 2 because of the stddev layer in Discriminator"
    assert fid_num % (val_batch_size*num_gpus) == 0, "Number of FID samples should be divisible with val_batch*gpus"
    if (use_ratio_loss):
        assert mixing_method is 'blender', "If ratio loss is used, blender net should be used instead of others"
    if mixing_method is 'encoder_in':
        assert config.repeat_w is False, "When styles are put inside the Encoder, there should be no broadcasting"

def infer_configs(config):
    #Intepret Other Configs Automatically
    config.wp_count = int( (math.log2(config.resolution)-1) * 2)
    config.z_count = 1
    config.create_mixing_network = config.mixing_method == 'fc'
    config.loss.hfgi_loss_kwargs.use = (config.encoder_type == 'hfgi') and (config.loss.hfgi_loss_kwargs.ada_lw > 0)
