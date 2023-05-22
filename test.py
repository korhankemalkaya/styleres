# python3.7
"""Main function for model inference."""

import os.path
import os
import shutil
import argparse
import math

import torch
import torch.distributed as dist

import runners
from utils.logger import build_logger
from utils.misc import init_dist
from utils.misc import DictAction, parse_config, update_config, infer_configs

from models import build_perceptual


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run model inference.')
    parser.add_argument('config', type=str,
                        help='Path to the inference configuration.')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='The work directory to save logs and checkpoints.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the checkpoint to load. (default: '
                             '%(default)s)')
    parser.add_argument('--synthesis_num', type=int, default=1,
                        help='Number of samples to synthesize. Set as 0 to '
                             'disable synthesis. (default: %(default)s)')
    parser.add_argument('--fid_num', type=int, default=0,
                        help='Number of samples to compute FID. Set as 0 to '
                             'disable FID test. (default: %(default)s)')
    parser.add_argument('--use_torchvision', action='store_true',
                        help='Wether to use the Inception model from '
                             '`torchvision` to compute FID. (default: False)')
    parser.add_argument('--launcher', type=str, default='pytorch',
                        choices=['pytorch', 'slurm'],
                        help='Launcher type. (default: %(default)s)')
    parser.add_argument('--edit_cfg', type=str, default='configs/edit_configs.py',
                        help='Launcher type.')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Backend for distributed launcher. (default: '
                             '%(default)s)')
    parser.add_argument('--rank', type=int, default=-1,
                        help='Node rank for distributed running. (default: '
                             '%(default)s)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Rank of the current node. (default: %(default)s)')
    parser.add_argument('--optimize_latent', action="store_true", default=False,
                        help='Whether to optimize latent code (default: %(default)s)')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='arguments in dict')
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments.
    args = parse_args()

    # Parse configurations.
    config = parse_config(args.config)
    config = update_config(config, args.options)
    edit_cfg = parse_config(args.edit_cfg)
    edit_cfg = update_config(edit_cfg, args.options)
    config.work_dir = args.work_dir
    config.checkpoint = args.checkpoint
    config.launcher = args.launcher
    config.backend = args.backend
    if not os.path.isfile(config.checkpoint):
        raise FileNotFoundError(f'Checkpoint file `{config.checkpoint}` is '
                                f'missing!')

    # Set CUDNN.
    config.cudnn_benchmark = config.get('cudnn_benchmark', True)
    config.cudnn_deterministic = config.get('cudnn_deterministic', False)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic

    # Setting for launcher.
    config.is_distributed = True
    init_dist(config.launcher, backend=config.backend)
    config.num_gpus = dist.get_world_size()
    infer_configs(config)

    # Setup logger.
    if dist.get_rank() == 0:
        logger_type = config.get('logger_type', 'normal')
        logger = build_logger(logger_type, work_dir=config.work_dir)
        shutil.copy(args.config, os.path.join(config.work_dir, 'config.py'))
        commit_id = os.popen('git rev-parse HEAD').readline()
        logger.info(f'Commit ID: {commit_id}')
    else:
        logger = build_logger('dumb', work_dir=config.work_dir)

    # Start inference.
    runner = getattr(runners, config.runner_type)(config, logger)
    runner.load(filepath=config.checkpoint,
                running_metadata=False,
                learning_rate=False,
                optimizer=False,
                running_stats=False)

    if (args.optimize_latent):
        runner.create_perceptual()
    use_meanw = runner.create_meanw()
    if use_meanw:
        runner.meanw = torch.from_numpy(runner.meanw).to(device=torch.cuda.current_device())
    if args.synthesis_num > 0:
        num = args.synthesis_num
        logger.print()
        logger.info(f'Synthesizing images ...')
  
        #runner.grad_edit('gender', 1, dataset = '/media/hdd2/adundar/hamza/genforce/data/temp/smile_with_original')
        #runner.save_edited_images('tanned', 0, dataset = '/media/hdd2/adundar/hamza/genforce/data/84')
        #runner.save_edited_images('inversion', 0, dataset = '/media/hdd2/adundar/hamza/genforce/data/395')
        
        #runner.save_edited_images('pose', 2, dataset = 'data/pose_center_original')

        runner.save_edited_images( edit_cfg)

        # runner.save_edited_images('inversion', 0)
        # runner.save_edited_images('smile', -3)
        # runner.save_edited_images('smile', 3, dataset = 'data/metfaces/images')

        logger.info(f'Finish synthesizing {num} images.')

    if args.fid_num > 0:
        num = args.fid_num
        logger.print()
        logger.info(f'Testing FID ...')
        fid_value = runner.fid(num, align_tf=not args.use_torchvision)
        logger.info(f'Finish testing FID on {num} samples. '
                    f'The result is {fid_value:.6f}.')


if __name__ == '__main__':
    main()
