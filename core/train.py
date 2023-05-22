from utils import get_data_iters, prepare_sub_folder, prepare_eval_folders, write_loss, get_config, write_2images, write_fid
import os
import sys
sys.path.append(os.getcwd())
from eval_utils.eval_processors import get_img_names
from fid_utils import save_real_images, eval_training
import argparse
from trainer import HiSD_Trainer
import torch
import tensorboardX
import shutil
import random

def read_img_list(img_list_file):
    with open(img_list_file) as img_list:
        lines = [line.split('\n')[0] for line in img_list.readlines()]
    return lines


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/main.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--train_name', type=str, help="Name for the model, to be used as output directory name")
parser.add_argument('--anno_path', type=str, help="Path for dataset annotation file including the file name")
parser.add_argument('--img_path', type=str, help="Path for image directory")
parser.add_argument('--eval_src', type=str)
parser.add_argument('--eval_real', type=str)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--gpus", nargs='+')
opts = parser.parse_args()

from torch.backends import cudnn

# For fast training
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
total_iterations = config['total_iterations']

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", opts.train_name))
output_directory = os.path.join(os.path.join(opts.output_path, opts.train_name), model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
eval_real_pos_path, eval_gen_pos_path, eval_real_neg_path, eval_gen_neg_path  = prepare_eval_folders(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Setup model
multi_gpus = len(opts.gpus) > 1
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opts.gpus)
trainer = HiSD_Trainer(config, multi_gpus=multi_gpus)
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

if multi_gpus:
    trainer.cuda(int(opts.gpus[0]))
    print("Using GPUs: %s" % str(opts.gpus))
    trainer.models= torch.nn.DataParallel(trainer.models, device_ids=[int(gpu) for gpu in opts.gpus])
else:
    trainer.cuda(int(opts.gpus[0]))

# Setup data loader
train_iters = get_data_iters(config, opts.gpus)
tags = list(range(len(train_iters)))

# Prepare for eval
"""
To filter the images for evaluation
mode: "r" for realism, "d" for disentanglement
gen_type: "r" for reference-guided, "l" for latent-guided
"""
eval_tag_name = "Smiling"
eval_id = 5
src_imgs_pos, real_imgs_pos = read_img_list(opts.eval_src), read_img_list(opts.eval_real)
src_imgs_neg, real_imgs_neg = read_img_list(opts.eval_real), read_img_list(opts.eval_src)
#src_imgs_d, real_imgs_d = get_img_names(opts.anno_path, mode="d", gen_type="l", target_feature=eval_tag_name, src_feature="Male", combined_feature="Young")
save_real_images(eval_real_pos_path, real_imgs_pos, config['crop_image_height']) # Images for positive
save_real_images(eval_real_neg_path, real_imgs_neg, config['crop_image_height']) # Images for negative
#save_real_images(eval_real_d_path, opts.img_path, real_imgs_d, config['crop_image_height']) # Images for disentanglement

import time
start = time.time()
while True:
    """
    i: tag
    j: source attribute, j_trg: target attribute
    x: image
    """
    i = random.sample(tags, 1)[0]
    j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 
    x = train_iters[i][j].next()
    train_iters[i][j].preload()

    G_adv, G_sty, G_rec, D_adv = trainer.update(x, i, j, j_trg)

    if (iterations + 1) % config['image_save_iter'] == 0:
        for i in range(len(train_iters)):
            j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 

            x = train_iters[i][j].next()
            x_trg = train_iters[i][j_trg].next()
            train_iters[i][j].preload()
            train_iters[i][j_trg].preload()
            trainer.models.gen.eval()
            test_image_outputs = trainer.sample(x, x_trg, j, j_trg, i)
            trainer.models.gen.train()
            write_2images(test_image_outputs,
                          config['batch_size'], 
                          image_directory, 'sample_%08d_%s_%s_to_%s' % (iterations + 1, config['tags'][i]['name'], config['tags'][i]['attributes'][j]['name'], config['tags'][i]['attributes'][j_trg]['name']))
    
    torch.cuda.synchronize()

    if (iterations + 1) % config['log_iter'] == 0:
        write_loss(iterations, trainer, train_writer)
        now = time.time()
        print(f"[#{iterations + 1:06d}|{total_iterations:d}] {now - start:5.2f}s")
        start = now

    if (iterations + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(checkpoint_directory, iterations)
        trainer.models.gen.eval()
        fid_pos = eval_training(src_imgs_pos, eval_real_pos_path, eval_gen_pos_path, config['crop_image_height'], trainer, 0)
        write_fid(iterations, "pos", fid_pos, train_writer)
        fid_neg = eval_training(src_imgs_neg, eval_real_neg_path, eval_gen_neg_path, config['crop_image_height'], trainer, 1)
        write_fid(iterations, "neg", fid_neg, train_writer)
        trainer.models.gen.train()


    if (iterations + 1) == total_iterations:
        print('Finish training!')
        exit(0)

    iterations += 1


