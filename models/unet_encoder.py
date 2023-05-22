import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import BasicBlock, NormLayer, ActivationLayer

_FINAL_RES = 4
class MiddleBlock(nn.Module):
    def __init__(self, features, middle_features, dropout=True, dropout_in_eval=True, activation='lrelu'):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(features, middle_features, bias=True), 
                                ActivationLayer(fn=activation))
        self.fc2 = nn.Sequential(nn.Linear(middle_features, features, bias=True), 
                                ActivationLayer(fn=activation))
        self.dropout = dropout                      #Whether to use dropout
        self.dropout_in_eval = dropout_in_eval      #Whether to keep dropout in eval mode. Comodgan keeps it.
        if self.dropout and self.dropout_in_eval == False:  
            self.drop = nn.Dropout(p=0.5, inplace=False)

    def _impl_dropout(self, x):
        if self.dropout:
            if self.dropout_in_eval:
                x = F.dropout(x, p=0.5, training=True, inplace=False) #Dropout is always used
            else:
                x = self.drop(x)
        return x
    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, c*w*h)
        x = self.fc1(x)
        x = self._impl_dropout(x)
        x = self.fc2(x)
        return x.view(b, c, w, h)

class StyleBlock(nn.Module):
    def __init__(self, in_channels, res, dropout=True, dropout_in_eval=True, use_blender=True, activation_fn='lrelu'):
        super().__init__()
        self.use_blender = use_blender
        self.shape = 512
        self.max_res = 4
        self.max_resolution = int(pow(self.max_res,2))
        
        if res > self.max_res:
            self.avg_pool = nn.AdaptiveAvgPool2d((self.max_resolution,self.max_resolution ) )
        if in_channels != self.shape:
            #self.conv = Conv2dLayer(in_channels, self.shape, kernel_size=1, bias=True,)
            self.conv = nn.Sequential( nn.Conv2d(in_channels, self.shape, kernel_size=1, padding=0, bias=True),
                                      ActivationLayer(fn=activation_fn))

        res = self.max_res if res > self.max_res else res
        self.style_blocks = nn.ModuleList()
        for i in range(res, 1, -1):
            layer = nn.Sequential(nn.Conv2d(self.shape, self.shape, kernel_size=3,bias=True,padding=1,stride=2),
                                 ActivationLayer(fn=activation_fn))
            #self.style_blocks.append( Conv2dLayer(self.shape, self.shape, kernel_size=3, bias=True, down=2, activation=activation))
            self.style_blocks.append(layer)

        #self.fc_layer = FullyConnectedLayer(self.shape*4, self.shape*2, bias=False, activation='linear')
        self.fc_layer = nn.Linear(self.shape*4, self.shape*2, bias=True)
        if self.use_blender:
            self.fc_layer2 = nn.Linear(self.shape*4, self.shape*2, bias=True)
            self.sigmoid = nn.Sigmoid()

        self.dropout = dropout                      #Whether to use dropout
        self.dropout_in_eval = dropout_in_eval      #Whether to keep dropout in eval mode. Comodgan keeps it.
        if self.dropout and self.dropout_in_eval == False:  
            self.drop = nn.Dropout(p=0.5, inplace=False)
    
    def _impl_dropout (self, x):
        if self.dropout:
            if self.dropout_in_eval:
                x = F.dropout(x, p=0.5, training=True, inplace=False) #Dropout is always used
            else:
                x = self.drop(x)
        return x

    def forward(self, x):
        if x.shape[-1] > self.max_resolution:
            x = self.avg_pool(x)
        if x.shape[1] != self.shape:
            x = self.conv(x)
        
        for block in self.style_blocks:
            x = block(x)
        b, c, w, h = x.shape
        x = x.view(b, c*w*h)
        x = self._impl_dropout(x)
        wp_enc = self.fc_layer(x)
        blender = None
        if self.use_blender:
            blender = self.fc_layer2(x)
            blender = self.sigmoid(blender)       
        return wp_enc, blender
        
class UpBlock(nn.Module):
    def __init__(self, in_channels,out_channels, res, upsample, dropout=True, dropout_in_eval=True, 
                norm_fn=None, activation_fn='lrelu', use_blender=True):
        super().__init__()
        self.upsample = upsample
        use_bias = norm_fn == None
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1,bias=use_bias),
                                 #NormLayer(fn=norm_fn, in_channels=in_channels),
                                 ActivationLayer(fn=activation_fn))
        self.style = StyleBlock( in_channels, res,  dropout=dropout, dropout_in_eval=dropout_in_eval, use_blender=use_blender)
        self.conv_out = None
        if in_channels != out_channels:
            self.conv_out = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,bias=True),
                                 #NormLayer(fn=norm_fn, in_channels=out_channels),
                                ActivationLayer(fn=activation_fn)
                                )

    def forward(self, x):
        x = self.conv_in(x)
        wp_enc, blender = self.style(x)
        if self.conv_out != None:
            x = self.conv_out(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x, wp_enc, blender

class UnetEncoder(nn.Module):
    arch_settings = {
        #Resolution Decreases as: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4. Last downsampling only includes one block
        26: (BasicBlock,  [2, 2, 2, 2, 2, 2]),
        42: (BasicBlock,  [3, 4, 6, 3, 2, 2]),
    }
    def __init__(self,
                 resolution,
                 latent_dim,
                 image_channels=4, #RGB + Mask
                 network_depth=26,
                 inplanes=64,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_fn=None,
                 activation_fn='lrelu',
                 downsample_fn='max_pool',
                 max_channels=512,
                 dropout = True,
                 dropout_in_eval = True,
                 use_blender='blender'):
        super().__init__()

        if network_depth not in self.arch_settings:
            raise ValueError(f'Invalid network depth: `{network_depth}`!\n'
                             f'Options allowed: '
                             f'{list(self.arch_settings.keys())}.')
        if isinstance(latent_dim, int):
            latent_dim = [latent_dim]
        assert isinstance(latent_dim, (list, tuple))

        self.resolution = resolution
        self.latent_dim = latent_dim
        self.image_channels = image_channels #RGB + Mask
        self.inplanes = inplanes
        self.network_depth = network_depth
        self.groups = groups
        self.dilation = 1
        self.base_width = width_per_group
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if norm_fn == None:
            use_bias = True
        else:
            use_bias = False
        self.norm_fn = norm_fn
        self.use_bias = use_bias
        self.activation_fn = activation_fn
        self.dowsample_fn = downsample_fn
        self.max_channels = max_channels
        use_blender= use_blender == 'blender'
        self.use_blender = use_blender

        block_fn, num_blocks_per_stage = self.arch_settings[network_depth]
        self.reslog2 = int(math.log2(resolution))
        self.num_stages = int(np.log2(resolution // _FINAL_RES))
        self.num_stages += 1
        for i in range(6, self.num_stages):
            num_blocks_per_stage.append(1)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * self.num_stages

        # Backbone.
        self.conv1 = nn.Conv2d(in_channels=image_channels,
                            out_channels=self.inplanes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=self.use_bias)
        self.norm = NormLayer(fn=self.norm_fn, in_channels=self.inplanes)
        self.activation = ActivationLayer(fn=self.activation_fn)

        self.stage_channels = [self.inplanes]
        self.basechannels = self.inplanes 
        for i in range(1, self.num_stages + 1):
            channels = min(self.max_channels, self.basechannels * (2 ** (i - 1)))
            num_blocks = num_blocks_per_stage[i - 1]
            stride = 1 if i == 1 else 2
            dilate = replace_stride_with_dilation[i - 1]
            self.add_module(f'layer{i}',
                            self._make_stage(block_fn=block_fn,
                                             planes=channels,
                                             num_blocks=num_blocks,
                                             stride=stride,
                                             dilate=dilate))
            self.stage_channels.append(channels)

        #Middle FC layers
        features = self.stage_channels[-1] * 4 * 4
        middle_features = 1024
        self.middle_layer = MiddleBlock(features, middle_features, dropout=dropout, 
                                dropout_in_eval=dropout_in_eval, activation=self.activation_fn)

        #Up Layers
        self.upblocks = nn.ModuleList()
        self.stage_channels = self.stage_channels[::-1] #Reverse the array
        
        for i in range(2, self.reslog2+1):
            upsample = i != self.reslog2
            self.upblocks.append( UpBlock( in_channels= self.stage_channels[i-2], out_channels= self.stage_channels[i-1] ,
                            res=i, norm_fn=norm_fn, activation_fn=activation_fn,
                            use_blender=use_blender, upsample=upsample))


    def _make_stage(self, block_fn, planes, num_blocks, stride=1, dilate=False):
        use_bias = self.use_bias
        downsample_fn = self.dowsample_fn
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            if (downsample_fn == 'strided'):
                downsample = nn.Sequential (nn.Conv2d(in_channels=self.inplanes,
                                        out_channels=planes * block_fn.expansion,
                                        kernel_size=1,
                                        stride=stride,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=False)
                            )
            elif(downsample_fn == 'max_pool' or downsample_fn == 'avg_pool'):
                downsample = nn.Sequential (nn.Conv2d(in_channels=self.inplanes,
                                        out_channels=planes * block_fn.expansion,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=False)
                            )

        blocks = []
        blocks.append(block_fn(inplanes=self.inplanes,
                               planes=planes,
                               base_width=self.base_width,
                               stride=stride,
                               groups=self.groups,
                               dilation=previous_dilation,
                               norm_fn = self.norm_fn,
                               use_bias = self.use_bias,
                               activation_fn = self.activation_fn,
                               downsample_fn = downsample_fn,
                               downsample=downsample))
        self.inplanes = planes * block_fn.expansion
        for _ in range(1, num_blocks):
            blocks.append(block_fn(inplanes=self.inplanes,
                                   planes=planes,
                                   base_width=self.base_width,
                                   stride=1,
                                   groups=self.groups,
                                   dilation=self.dilation,
                                   norm_fn = self.norm_fn,
                                   use_bias = self.use_bias,
                                   activation_fn = self.activation_fn,
                                   downsample_fn = downsample_fn,
                                   downsample=None))

        return nn.Sequential(*blocks)

    def forward(self, reals, valids):
        if (type(valids) is int):
            b, c, h, w = reals.size()
            if (valids == 1):
                valids = torch.ones(size=(b,1,h,w), dtype=reals.dtype, device=reals.device)
            elif (valids == 0):
                valids = torch.zeros(size=(b,1,h,w), dtype=reals.dtype, device=reals.device)
            else:
                raise "Valid integer can be 1 or 0"

        rgb_masked = torch.cat((reals*valids, valids), dim=1)
        x = self.conv1(rgb_masked)
        x = self.norm(x, rgb_masked)
        x = self.activation(x)

        E_feats = []
        for i in range(1, self.num_stages + 1):
            layers = getattr(self, f'layer{i}')
            for layer in layers:
                x = layer(x, rgb_masked)
            E_feats.append(x)
        x = self.middle_layer(x)

        wp_list = []
        blender_list = []
        E_feats = E_feats[::-1] #Reverse the list
        for i in range( len(self.upblocks)):
            x, wp_enc, blender = self.upblocks[i](E_feats[i] + x)
            wp_list.extend(torch.split(wp_enc, 512, 1))
            if self.use_blender:
                blender_list.extend(torch.split(blender, 512, dim=1))
        wp_enc = torch.stack(wp_list, dim=1)
        blender = torch.stack(blender_list, dim=1) if self.use_blender else None
        
        return wp_enc, blender


        
        
        

        

        
