#!/bin/bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
PORT=${PORT:-29500}

python -m torch.distributed.launch \
       --nproc_per_node=${GPUS} \
       --master_port=${PORT} \
       ./train.py ${CONFIG} \
           --work_dir ${WORK_DIR} \
           --launcher="pytorch" \
           ${@:4}

# Example run with command line          
#CUDA_VISIBLE_DEVICES=7 scripts/dist_train.sh 1 configs/stylegan_ffhq256_encoder_y.py work_dirs/debug