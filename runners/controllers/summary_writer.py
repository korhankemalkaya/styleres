#Defines a singleton class for Tensorboard
#Set up it after setting up controllers
#Call it everywhere
#Do not directly import this class, otherwise writer will be None. Only import necessary functions.

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore TF warning.
import os
from torch.utils.tensorboard import SummaryWriter

tensorboard_writer = None

def setup(work_dir, rank):
    if (rank != 0):
        return 
    global tensorboard_writer

    event_dir = os.path.join(work_dir, 'events')
    os.makedirs(event_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=event_dir)

def end(rank):
    if (rank != 0):
        return
    global tensorboard_writer
    tensorboard_writer.close()

def log_data(log_data, runner):
    if (runner.rank != 0):
        return
    global tensorboard_writer

    for name, value in log_data.items():
        if name in ['data_time', 'iter_time', 'run_time']:
            continue
        if name.startswith('loss_'):
            tensorboard_writer.add_scalar(
                name.replace('loss_', 'loss/'), value, global_step=runner.iter)
        elif name.startswith('lr_'):
            tensorboard_writer.add_scalar(
                name.replace('lr_', 'learning_rate/'), value, global_step=runner.iter)
        elif name.startswith('fid'):
            if( value != 0):
                tensorboard_writer.add_scalar(name, value, global_step=runner.iter)
        elif name.startswith('metric'):
            if( value != 0):
                tensorboard_writer.add_scalar(name, value, global_step=runner.iter)
        else:
            tensorboard_writer.add_scalar(name, value, global_step=runner.iter)

def log_image(name="", grid=None, iter= -1):
    global tensorboard_writer
    tensorboard_writer.add_image(name, grid, iter)

