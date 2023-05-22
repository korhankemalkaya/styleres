# python3.7
"""Contains the running controller to handle checkpoints."""

import os.path

from .base_controller import BaseController

__all__ = ['Checkpointer']

class Checkpointer(BaseController):
    """Defines the running controller to handle checkpoints.

    This controller is used to save and load checkpoints.

    NOTE: This controller is set to `LOW` priority by default and will only be
    executed on the master worker.
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'LOW')
        config.setdefault('master_only', True)
        super().__init__(config)

        self._save_dir = config.get('checkpoint_dir', None)
        self._save_running_metadata = config.get('save_running_metadata', True)
        self._save_learning_rate = config.get('save_learning_rate', True)
        self._save_optimizer = config.get('save_optimizer', True)
        self._save_running_stats = config.get('save_running_stats', False)
        self.ignore_list_latest = ['encoder'] #['generator', 'generator_smooth']
        self.ignore_list_best =  ['encoder' ,'discriminator'] # 'generator', 'generator_smooth'

        self.best_fid = config.get('initial_fid', 1000)
        self.save_best = False
        self.save_iter = False

    def execute_after_iteration(self, runner):
        save_dir = self._save_dir or runner.work_dir
        #save_filename = f'checkpoint_iter{runner.iter:06d}.pth'
        if (self.save_best):
            save_filename = f'best.pth'
            runner.save(filepath=os.path.join(save_dir, save_filename),
                        running_metadata=self._save_running_metadata,
                        learning_rate=False,
                        optimizer=False,
                        running_stats=False,
                        ignore_models = self.ignore_list_best)
            self.save_best = False

        if (self.save_iter):
            save_filename = f'latest.pth'
            runner.save(filepath=os.path.join(save_dir, save_filename),
                        running_metadata=self._save_running_metadata,
                        learning_rate=self._save_learning_rate,
                        optimizer=self._save_optimizer,
                        running_stats=self._save_running_stats,
                        ignore_models = self.ignore_list_latest)
            self.save_iter = False
        

    def is_executable(self, runner):
        if self.master_only and runner.rank != 0:
            return False

        if self.first_iter and runner.iter - runner.start_iter == 1:
            self.save_iter = True
        elif runner.iter == runner.total_iters:
            self.save_iter = True
        elif self.every_n_iters > 0 and runner.iter % self.every_n_iters == 0:
            self.save_iter = True
        elif runner.save_now:
            self.save_iter = True

        epoch_to_iter = runner.convert_epoch_to_iter(self.every_n_epochs)
        if self.every_n_epochs > 0 and runner.iter % epoch_to_iter == 0:
            self.save_iter = True
        
        #current_fid = runner.running_stats.stats_pool['fid'].val
        smile_remove_fid = runner.running_stats.stats_pool['fid_smile_remove'].val
        smile_addition_fid = runner.running_stats.stats_pool['fid_smile_addition'].val
        current_fid = (smile_remove_fid + smile_addition_fid) / 2
        if (current_fid != 0 and current_fid < self.best_fid):
            self.best_fid = current_fid 
            self.save_best = True

        return self.save_iter or self.save_best
