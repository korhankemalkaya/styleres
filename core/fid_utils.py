import os
import numpy as np
import torch
from PIL import Image
from core.trainer import HiSD_Trainer
import torchvision.utils as vutils
from torchvision import transforms
from eval_utils.fid import calculate_fid_given_paths
from eval_utils.eval_processors import get_img_names

def save_real_images(save_path, img_list, img_res):
    image_transform = transforms.Compose([transforms.Resize((img_res, img_res)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    for img_name in img_list:
        img = image_transform(Image.open(img_name).convert("RGB")).unsqueeze(0)
        file_name = img_name.split(os.sep)[-1]
        vutils.save_image((img + 1) / 2, os.path.join(save_path, file_name))

@torch.no_grad()
def infer_l_eval_set(img_list, trainer, output_path, img_res, tag, attribute):
    image_transform = transforms.Compose([transforms.Resize((img_res, img_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    for img_name in img_list:
        x = image_transform(Image.open(img_name).convert("RGB")).unsqueeze(0).cuda()
        encoded_features, skip_conn = trainer.models.gen.encode(x) # Encoding step
        alpha_x = trainer.models.gen.extract(encoded_features, tag) # Alpha value for the input
        z = torch.rand(1, 1).cuda() # Random uniform sampling
        alpha = trainer.models.gen.map(z, tag, attribute) # Transform randomly sampled z
        shift_amount = alpha - alpha_x
        encoded_features = trainer.models.gen.translate(encoded_features, tag, shift_amount)
        x_out = trainer.models.gen.decode(encoded_features, skip_conn)
        img = img_name.split(os.sep)[-1]
        vutils.save_image((x_out + 1) / 2, os.path.join(output_path, img))

@torch.no_grad()
def infer_r_eval_set(img_path, img_list, ref_list, trainer, output_path, img_res, tag):
    image_transform = transforms.Compose([transforms.Resize((img_res, img_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    reference_idx = np.random.permutation(len(ref_list))

    for img_name_idx in range(len(img_list)):
        full_img_path = os.path.join(img_path, img_list[img_name_idx])
        x = image_transform(Image.open(full_img_path).convert("RGB")).unsqueeze(0).cuda()
        encoded_features, skip_conn = trainer.models.gen.encode(x)
        alpha_x = trainer.models.gen.extract(encoded_features, tag)
        ref_img_path = os.path.join(img_path, ref_list[reference_idx[img_name_idx % len(reference_idx % len(reference_idx))]])
        ref_img = image_transform(Image.open(ref_img_path).convert("RGB")).unsqueeze(0).cuda()
        ref_encoding, _ = trainer.models.gen.encode(ref_img)
        alpha = trainer.models.gen.extract(ref_encoding, tag)
        shift_amount = alpha - alpha_x
        encoded_features = trainer.models.gen.translate(encoded_features, tag, shift_amount)
        x_out = trainer.models.gen.decode(encoded_features, skip_conn)
        vutils.save_image((x_out + 1) / 2, os.path.join(output_path, img_list[img_name_idx]))


def eval_training(src_imgs_r, eval_real_r_path, eval_gen_r_path, img_res, trainer, attr):
    tag, attr = 5, attr
    infer_l_eval_set(src_imgs_r, trainer, eval_gen_r_path, img_res, tag, attr) # Inference for realism
    #infer_l_eval_set(src_path, src_imgs_d, trainer, eval_gen_d_path, img_res, tag, attr) # Inference for disentanglement
    fid_r = calculate_fid_given_paths([eval_real_r_path, eval_gen_r_path], img_size=img_res, enable_tqdm=False)
    #fid_d = calculate_fid_given_paths([eval_real_d_path, eval_gen_d_path], img_size=img_res, enable_tqdm=False)
    fid_type = ["pos", "neg"]
    print("FID-{}: {}".format(fid_type[attr], fid_r))
    return fid_r
