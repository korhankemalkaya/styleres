export CUDA_VISIBLE_DEVICES=0

#Compare real smiles and fake smiles 
REAL="/media/hdd2/adundar/hamza/genforce/data/temp/smile_with_original"
FAKE="/media/hdd2/adundar/korhan/styleres/work_dirs/outputs/interfacegan_smile_3"
python fid.py --paths $REAL $FAKE

#Compare real non-smiles and fake none_smiles 
REAL="/media/hdd2/adundar/hamza/genforce/data/temp/smile_without_original"
FAKE="/media/hdd2/adundar/korhan/styleres/work_dirs/outputs/interfacegan_smile_-3"
python fid.py --paths $REAL $FAKE

REAL="/media/hdd2/adundar/hamza/genforce/data/temp/images256"
FAKE="/media/hdd2/adundar/korhan/styleres/work_dirs/outputs/inversion"
python fid.py --paths $REAL $FAKE
