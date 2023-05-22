# python3.7
"""Contains the running controller for evaluation."""

import os.path
import time

from .base_controller import BaseController
from ..misc import format_time

__all__ = ['FIDEvaluator']


class FIDEvaluator(BaseController):
    """Defines the running controller for evaluation.

    This controller is used to evalute the GAN model using FID metric.

    NOTE: The controller is set to `MEDIUM` priority by default.
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'MEDIUM')
        super().__init__(config)

        self.num = config.get('num', 50000)
        self.ignore_cache = config.get('ignore_cache', False)
        self.align_tf = config.get('align_tf', True)
        self.file = None

    def setup(self, runner):
        assert hasattr(runner, 'fid')
        file_path = os.path.join(runner.work_dir, f'metric_fid{self.num}.txt')
        fid_smile_remove_path = os.path.join(runner.work_dir, f'fid_smile_remove.txt')
        fid_smile_addition_path = os.path.join(runner.work_dir, f'fid_smile_add.txt')

        self.metrics = ['mse', 'lpips', 'ssim']
        fpaths = []
        self.eval_files = {}
        for m in self.metrics:
            fpaths.append(os.path.join(runner.work_dir, f'metric_{m}.txt'))
        if runner.rank == 0:
            self.file = open(file_path, 'w')
            self.file_smile_remove = open(fid_smile_remove_path, 'w')
            self.file_smile_add = open(fid_smile_addition_path, 'w')

            for m, p in zip(self.metrics, fpaths):
                self.eval_files[m] = open(p, 'w')
        runner.running_stats.add(f'fid', log_format='.3e', log_name=f'fid', log_strategy='CURRENT')
        runner.running_stats.add(f'fid_smile_remove', log_format='.3e', log_name=f'fid_smile_remove', log_strategy='CURRENT')
        runner.running_stats.add(f'fid_smile_addition', log_format='.3e', log_name=f'fid_smile_addition', log_strategy='CURRENT')

        for m in self.metrics:
            runner.running_stats.add(f'metric_{m}', log_format='.3e', log_name=f'metric_{m}', log_strategy='CURRENT')

    def close(self, runner):
        if runner.rank == 0:
            self.file.close()

    def execute_after_iteration(self, runner):
        mode = runner.mode  # save runner mode.
        #Smile Remove Metric
        # start_time = time.time()
        # smile_remove_fid = runner.fid_attribute(10000, factor=-3)
        # duration_str = format_time(time.time() - start_time)
        # log_str = (f'FID-: {smile_remove_fid:.5f} at iter {runner.iter:06d} '
        #            f'({runner.seen_img / 1000:.1f} kimg). ({duration_str})')
        # runner.logger.info(log_str)
        # if runner.rank == 0:
        #     date = time.strftime("%Y-%m-%d %H:%M:%S")
        #     self.file_smile_remove.write(f'[{date}] {log_str}\n')
        #     self.file_smile_remove.flush()
        # runner.running_stats.update({f'fid_smile_remove': smile_remove_fid})

        # #Smile Addition Metric
        # start_time = time.time()
        # smile_add_fid = runner.fid_attribute(10000, factor=3)
        # duration_str = format_time(time.time() - start_time)
        # log_str = (f'FID+: {smile_add_fid:.5f} at iter {runner.iter:06d} '
        #            f'({runner.seen_img / 1000:.1f} kimg). ({duration_str})')
        # runner.logger.info(log_str)
        # if runner.rank == 0:
        #     date = time.strftime("%Y-%m-%d %H:%M:%S")
        #     self.file_smile_add.write(f'[{date}] {log_str}\n')
        #     self.file_smile_add.flush()
        # runner.running_stats.update({f'fid_smile_addition': smile_add_fid})


        #FID Metric
        start_time = time.time()
        fid_value = runner.fid()
        if runner.rank == 0:
            duration_str = format_time(time.time() - start_time)
            log_str = (f'FID: {fid_value:.5f} at iter {runner.iter:06d} '
            f'({runner.seen_img / 1000:.1f} kimg). ({duration_str})')
            runner.logger.info(log_str)
            date = time.strftime("%Y-%m-%d %H:%M:%S")
            self.file.write(f'[{date}] {log_str}\n')
            self.file.flush()
        runner.running_stats.update({f'fid': fid_value})
        runner.set_mode(mode)  # restore runner mode.
