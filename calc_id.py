import argparse
import os
import torch
from tqdm import tqdm
from arcface.arcface import ArcFace
from torchvision import transforms
from PIL import Image
import numpy as np

img_res = 256

input_transform = transforms.Compose([transforms.Resize((img_res, img_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str)
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--model_out_path", type=str)
args = parser.parse_args()

model = ArcFace(args.ckpt_path)
model.cuda()
model.eval()

print("INFO: ArcFace loaded")

folder_values = []

#out_path = os.path.join(args.model_out_path, f"exp_{exp_count}")
out_path = args.model_out_path
generated_img_list = os.listdir(out_path)

img_idxs = [int(img.split(sep=".")[0]) for img in generated_img_list]

similarity_values = torch.zeros(len(generated_img_list)).cuda()
with torch.no_grad():
    for img_idx in tqdm(range(len(generated_img_list))):
        real_img_path = os.path.join(args.img_path, "{}.jpg".format(img_idxs[img_idx]))
        real_img = input_transform(Image.open(real_img_path).convert("RGB")).unsqueeze(0).cuda()
        gen_img_path = os.path.join(out_path, generated_img_list[img_idx])
        gen_img = input_transform(Image.open(gen_img_path).convert("RGB")).unsqueeze(0).cuda()
        similarity = model(real_img, gen_img)
        similarity_values[img_idx] = similarity.squeeze(0)
        
    mean_sim = torch.mean(similarity_values)
    print("Mean similarity: {}".format(mean_sim))

