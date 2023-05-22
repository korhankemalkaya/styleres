import sys
import os

from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
from utils import weights_init
import math

RGB_DIM = 3

##################################################################################
# Discriminator
##################################################################################

class Dis(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.tags = hyperparameters['tags']
        channels = hyperparameters['discriminators']['channels']

        self.conv = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            #nn.AdaptiveAvgPool2d(1),
        )
        
        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels[-1] + 
            # ALI part which is not shown in the original submission but help disentangle the extracted style. 
            # hyperparameters['style_dim'] +
            # 1 dimension for alpha
            1,
            # One for translated, one for cycle. Eq.4
            len(self.tags[i]['attributes'] * 2), 1, 1, 0),
        ) for i in range(len(self.tags))])
        

    def forward(self, x, s, i):
        f = self.conv(x)
        fs = torch.cat([f, tile_like(s, f)], 1)
        return self.fcs[i](fs).view(f.size(0), 2, -1)
        

    def calc_dis_loss_real(self, x, s, i, j):
        loss = 0
        x = x.requires_grad_()
        out = self.forward(x, s, i)[:, :, j]
        loss += F.relu(1 - out[:, 0]).mean()
        loss += F.relu(1 - out[:, 1]).mean()
        loss += self.compute_grad2(out[:, 0], x)
        loss += self.compute_grad2(out[:, 1], x)
        return loss
    
    def calc_dis_loss_fake_trg(self, x, s, i, j):
        out = self.forward(x, s, i)[:, :, j]
        loss = F.relu(1 + out[:, 0]).mean()
        return loss
    
    def calc_dis_loss_fake_cyc(self, x, s, i, j):
        out = self.forward(x, s, i)[:, :, j]
        loss = F.relu(1 + out[:, 1]).mean()
        return loss

    def calc_gen_loss_real(self, x, s, i, j):
        loss = 0
        out = self.forward(x, s, i)[:, :, j]
        loss += out[:, 0].mean()
        loss += out[:, 1].mean()
        return loss

    def calc_gen_loss_fake_trg(self, x, s, i, j):
        out = self.forward(x, s, i)[:, :, j]
        loss = - out[:, 0].mean()
        return loss

    def calc_gen_loss_fake_cyc(self, x, s, i, j):
        out = self.forward(x, s, i)[:, :, j]
        loss = - out[:, 1].mean()
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
         )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg.mean()

## translator
class Translator(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        channels = hyperparameters['translators']['channels']
        self.model = nn.Sequential(
            nn.Conv2d(hyperparameters['encoder']['channels'][-1], channels[0], 1, 1, 0),
            *[MiddleBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.style_to_params = nn.Linear(hyperparameters['style_dim'], self.get_num_adain_params(self.model))

        self.features = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
        )

        self.masks = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, e, s):
        p = self.style_to_params(s)
        self.assign_adain_params(p, self.model)

        mid = self.model(e)
        f = self.features(mid)
        m = self.masks(mid)

        return f * m + e * (1 - m)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m.weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features, 1) + 1
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2 * m.num_features
        return num_adain_params

##################################################################################
# Generator
##################################################################################

class Gen(nn.Module):
    def __init__(self, hyperparameters, random_init=True):
        super().__init__()
        self.tags = hyperparameters['tags']
        
        channels = hyperparameters['encoder']['channels']
        self.encoder = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlockIN(channels[i], channels[i+1]) for i in range(len(channels) - 2)],
            DownBlock(channels[len(channels) - 2], channels[len(channels) - 1]),
        )

        style_channels = channels[-1]

        channels = hyperparameters['decoder']['channels']
        
        self.decoder = nn.Sequential(
            UpBlock(channels[0], channels[1]),
            *[UpBlockIN(channels[i], channels[i + 1]) for i in range(1, len(channels) - 1)],
            nn.Conv2d(channels[-1], hyperparameters['input_dim'], 1, 1, 0)
        )

        self.encoder_map_dim = torch.tensor([style_channels, 
            hyperparameters["crop_image_height"] // 2**(len(channels) - 1),
            hyperparameters["crop_image_width"] // 2**(len(channels) - 1)]).cuda()
        
        self.enc_dim = torch.prod(self.encoder_map_dim) # Dimsnion of flattened direction
        self.latent_dim = hyperparameters['latent_dim']

        self.direction_matrix = nn.Parameter(data=(1.0 if random_init else 0.001) * torch.randn(size=(len(self.tags), self.latent_dim)), requires_grad=True)
        
        # To control the dimensions of the feature encoding
        self.vectorization_unit = nn.ModuleList([nn.Conv2d(self.enc_dim, self.latent_dim, 1, 1, 0), nn.Conv2d(self.latent_dim, self.enc_dim, 1, 1, 0)])

        self.mappers = nn.ModuleList([ShiftMapper(len(self.tags[i]["attributes"])) for i in range(len(self.tags))])
        
        reverse_encoder_channels = hyperparameters['encoder']['channels'][::-1]
        decoder_channels = hyperparameters['decoder']['channels']
        print(hyperparameters)
        print(hyperparameters['decoder'])
        print(hyperparameters['decoder']['channels'])

        # Currently not using attention on skip connections
        #mask_channel = hyperparameters['encoder']['channels'][2]
        #self.masks = AttentionBlk(mask_channel, mask_channel)

        self.mse_loss = nn.MSELoss()
        self.style_to_params = nn.Linear(style_channels, self.get_num_adain_params(self.decoder))

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m.weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features, 1) + 1
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2 * m.num_features
        return num_adain_params



    def encode(self, x):        
        e = x
        s = None
        for layer_idx in range(len(self.encoder)):
            e = self.encoder[layer_idx](e)
            if layer_idx == 2:
                s = e
        e_flat = e.reshape(e.shape[0], self.enc_dim, 1, 1)
        e = self.vectorization_unit[0](e_flat).reshape(e.shape[0], self.latent_dim)
        return e, s
        

    def decode(self, e, s):
        p = self.style_to_params(e)
        self.assign_adain_params(p, self.decoder)

        e = e.reshape(e.shape[0], self.latent_dim, 1, 1)
        x = self.vectorization_unit[1](e)
        x = x.reshape(e.shape[0], *self.encoder_map_dim)
        for layer_idx in range(len(self.decoder)):
            if layer_idx == (len(self.encoder) - 3):
                #attn_mask = self.masks(x)
                #x = x * attn_mask + s * (1 - attn_mask)
                #x = x + s * attn_mask
                # Currently not using attention on skip connections
                x = x + s
            x = self.decoder[layer_idx](x)
        return x


    # Changed Functions
    def extract(self, e, tag_idx):
        dir_vector = self.direction_matrix[tag_idx, :]
        # Dimension 0 is batch_size
        alpha = torch.mm(e, dir_vector.reshape(dir_vector.shape[0], 1)) / torch.dot(dir_vector, dir_vector)
        return alpha

    def map(self, alpha, i, j):
        return self.mappers[i](alpha, j)

    def translate(self, e, tag_idx, alpha):
        dir_vector = torch.mul(self.direction_matrix[tag_idx, :], alpha)
        return e + dir_vector

    def calc_sparsity_loss(self):
        return torch.mean(torch.abs(self.direction_matrix))

    def calc_disentanglement_loss(self, e, e_trg, tag):
        """
        Comapring the scales for unedited attributes for disetnaglement
        e: input image encoding
        e_trg: output image encoding
        tag: edited tag
        """
        target_tags = [i for i in range(len(self.tags)) if i != tag]
        preserved_directions = self.direction_matrix[target_tags]
        initial_scales = torch.tensordot(e, preserved_directions, dims=([1], [1]))
        final_scales = torch.tensordot(e_trg, preserved_directions, dims=([1], [1]))
        return F.l1_loss(final_scales, initial_scales)



# This block is not used currently
class AttentionBlk(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(AttentionBlk, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        return self.sigmoid(out)


class MapperBlock(nn.Module):
    # hidden_features: Number of neurons on hidden layer
    def __init__(self, hidden_features=256):
        super(MapperBlock, self).__init__()
        self.line_endpoints = nn.Parameter(data=torch.randn(2), requires_grad=True)

    def forward(self, z):
        """
        line_endpoints = start_of_line, end_of_line
        """
        return z * (self.line_endpoints[1] - self.line_endpoints[0]) + self.line_endpoints[0]


class ShiftMapper(nn.Module):
    def __init__(self, num_attributes):
        super(ShiftMapper, self).__init__()
        # This module defines the line endpoints for a given tag
        self.endpoints = nn.Parameter(data=torch.randn(num_attributes + 1), requires_grad=True)

    def forward(self, z, j):
        return z * (self.endpoints[j + 1] - self.endpoints[j]) + self.endpoints[j]


def direction_l2_norm(direction_matrix):
    epsilon = 1e-6 # Safety threshold for sqrt(0)
    norm = torch.pow(torch.sum(torch.pow(direction_matrix, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(direction_matrix)
    return torch.div(direction_matrix, norm)


##################################################################################
# Basic Blocks
##################################################################################

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(F.avg_pool2d(self.conv1(self.activ(x.clone())), 2)))
        out = residual + out
        return out / math.sqrt(2)

class DownBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(in_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(self.in2(F.avg_pool2d(self.conv1(self.activ(self.in1(x.clone()))), 2))))
        out = residual + out
        return out / math.sqrt(2)

class DownBlockBN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=in_dim)
        self.bn2 = nn.BatchNorm2d(num_features=in_dim)

        self.activ = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        res_branch = F.avg_pool2d(self.sc(x), 2)
        out = self.conv1(self.activ(self.bn1(x.clone())))
        out = self.conv2(self.activ(self.bn2(F.avg_pool2d(out, 2))))
        out = res_branch + out
        return out / math.sqrt(2)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.conv1(F.interpolate(self.activ(x.clone()), scale_factor=2, mode='nearest'))))
        out = residual + out
        return out / math.sqrt(2)

class UpBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1 = AdaptiveInstanceNorm2d(in_dim)
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.adain2(self.conv1(F.interpolate(self.activ(self.adain1(x.clone())), scale_factor=2, mode='nearest')))))
        out = residual + out
        return out / math.sqrt(2)

class UpBlockAdaIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1 = AdaptiveInstanceNorm2d(in_dim)
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):

        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.adain2(self.conv1(F.interpolate(self.activ(self.adain1(x.clone())), scale_factor=2, mode='nearest')))))
        out = residual + out
        return out / math.sqrt(2)




class UpBlockBN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=in_dim)
        self.bn2 = nn.BatchNorm2d(num_features=out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        res_branch = F.interpolate(self.sc(x), scale_factor=2, mode="nearest")
        out = F.interpolate(self.activ(self.bn1(x.clone())), scale_factor=2, mode="nearest")
        out = self.conv2(self.activ(self.bn2(self.conv1(out))))
        out = res_branch + out
        return out / math.sqrt(2)


class MiddleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1 = AdaptiveInstanceNorm2d(in_dim)
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = self.sc(x)
        out = self.conv2(self.activ(self.adain2(self.conv1(self.activ(self.adain1(x.clone()))))))
        out = residual + out
        return out / math.sqrt(2)

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear(self.activ(x))

##################################################################################
# Basic Modules and Functions
##################################################################################

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.bias = None
        self.weight = None

    def forward(self, x):
        assert self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

def tile_like(x, target):
    # make x is able to concat with target at dim 1.
    x = x.view(x.size(0), -1, 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x
