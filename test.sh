export CUDA_VISIBLE_DEVICES="3"
CONFIG="configs/stylegan_ffhq256_encoder_y.py"
WORKDIR="/media/hdd2/adundar/korhan/styleres/work_dirs/outputs"
CKPT="/media/hdd2/adundar/korhan/styleres/work_dirs/psp_reduced46/best.pth"

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 test.py $CONFIG --checkpoint $CKPT --work_dir $WORKDIR --launcher pytorch