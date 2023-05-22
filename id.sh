export CUDA_VISIBLE_DEVICES="0"

# ID +
IMG_PATH="/media/hdd2/adundar/hamza/genforce/data/temp/smile_without_original"
MODEL_OUT_PATH="/media/hdd2/adundar/korhan/styleres/work_dirs/outputs/interfacegan_smile_3"
MODEL_PATH="/media/hdd1/adundar/hamza/yusuf_hoca/CurricularFace_Backbone.pth"
python calc_id.py --img_path $IMG_PATH --ckpt_path $MODEL_PATH --model_out_path $MODEL_OUT_PATH

# ID -
IMG_PATH="/media/hdd2/adundar/hamza/genforce/data/temp/smile_with_original"
MODEL_OUT_PATH="/media/hdd2/adundar/korhan/styleres/work_dirs/outputs/interfacegan_smile_-3"
python calc_id.py --img_path $IMG_PATH --ckpt_path $MODEL_PATH --model_out_path $MODEL_OUT_PATH
