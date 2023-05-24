export CUDA_VISIBLE_DEVICES="1"
export TORCH_HOME="torch_home/"dd

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=29504 train.py configs/stylegan_ffhq256_encoder_y.py --work_dir work_dirs/wo_adv

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 train.py configs/stylegan_ffhq256_encoder_y.py --work_dir work_dirs/psp_reduced46
 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 train.py configs/stylegan_ffhq256_encoder_y.py --work_dir /home/korhan/styleres/work_dirs/psp_deformableconv