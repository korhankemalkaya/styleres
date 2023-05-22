import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.util import get_blocks, bottleneck_IR, bottleneck_IR_SE, EqualLinear

class WEncoder(nn.Module):
    def __init__(self, num_layers, mode='ir', out_res = 64):
        super(WEncoder, self).__init__()
        self.out_res = out_res
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.PReLU(64))
        self.output_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)
        self.style_count = 18

    def forward(self, x):
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        c = {   128: c0,
                64: c1,
                32: c2,
                16: c3
             }.get(self.out_res)
        return x.repeat(self.style_count, 1, 1).permute(1, 0, 2), c

class HyperStyle(nn.Module):
    def __init__(self, resolution, path, num_layers = 50, mode='ir_se', out_res=64) :
        super(HyperStyle, self).__init__( )
        self.out_res = out_res
        self.basic_encoder = WEncoder(num_layers, mode, out_res)
        ckpt = torch.load(path, map_location='cpu')
        self.latent_avg = ckpt['latent_avg'].cuda()
        ckpt = {k[k.find(".")+1:]: v for k, v in ckpt['state_dict'].items() if "decoder" not in k}
        self.basic_encoder.load_state_dict(ckpt, strict=True)
        self.freeze_basic_encoder()

    def freeze_basic_encoder(self):
        self.basic_encoder.eval()   #Basic Encoder always in eval mode.
        #No backprop to basic Encoder
        for param in self.basic_encoder.parameters():
            param.requires_grad = False 

    def forward(self, reals):
        self.freeze_basic_encoder()
        w, c = self.basic_encoder(reals)
        w = w + self.latent_avg
        highres_outs = {f"{self.out_res}x{self.out_res}": c} #{"gates": gates, "additions": additions}
        return w, highres_outs