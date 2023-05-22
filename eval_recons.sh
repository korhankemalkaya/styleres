
GTPATH="/media/hdd2/adundar/hamza/genforce/data/temp/images256"
DPATH="/media/hdd2/adundar/korhan/styleres/work_dirs/outputs/inversion"

python scripts/calc_losses_on_images.py --mode lpips --data_path $DPATH --gt_path $GTPATH
python scripts/calc_losses_on_images.py --mode ssim --data_path $DPATH --gt_path $GTPATH
