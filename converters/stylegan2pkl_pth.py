import os
import sys
import inspect

currentdir  = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import torch
from models.stylegan2_official import Generator
from models.stylegan2_official import Discriminator
import dnnlib
from legacy import load_network_pkl
from torch_utils import misc
PATH = 'dummy.pth'

G_kwargs=dict()
G_kwargs['fused_modconv_default'] = 'inference_only' # Speed up training by using regular convolutions instead of grouped
#G_kwargs['channel_base'] = 16384 #Uncomment  for 256x256 res
G_ema = Generator(z_dim=512, c_dim=0, w_dim=512, resolution=1024, img_channels=3, **G_kwargs)
G = Generator(z_dim=512, c_dim=0, w_dim=512, resolution=1024, img_channels=3, **G_kwargs)
D_kwargs=dict()
#D_kwargs['channel_base'] = 16384 #Uncomment  for 256x256 res
D = Discriminator(c_dim=0, img_resolution=1024, img_channels=3, **D_kwargs)

for param in G_ema.parameters():
	param.requires_grad = False
for param in G.parameters():
	param.requires_grad = False
for param in D.parameters():
	param.requires_grad = False

z = torch.randn([1, G.z_dim])
c = None              
img = G(z, c) 
network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl'
network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl'
network_pkl = '/media/hdd2/adundar/hamza/genforce/stylegan2-ffhq-1024x1024.pkl'
with dnnlib.util.open_url(network_pkl) as fp:
	networks = load_network_pkl(fp)
for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
	misc.copy_params_and_buffers(networks[name], module, require_all=False)

torch.save({'generator_smooth':G_ema.state_dict(),
		    'generator' : G.state_dict(),
		    'discriminator' : D.state_dict()
		    }, PATH)		   
ckpt = torch.load(PATH)
G_ema.load_state_dict(ckpt['generator_smooth'])
D.load_state_dict(ckpt['discriminator'])
print('success')