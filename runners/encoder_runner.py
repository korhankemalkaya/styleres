# python3.7
"""Contains the runner for Encoder."""

import torch.nn.functional as F
from copy import deepcopy

from .base_encoder_runner import BaseEncoderRunner

from .forwarder.train_forwarder import TrainForwarder
from .forwarder.eval_forwarder import EvalForwarder

__all__ = ['EncoderRunner']


class EncoderRunner(BaseEncoderRunner):
    """Defines the runner for Enccoder Training."""

    def build_models(self):
        super().build_models()
        self.train_forwarder = TrainForwarder()
        self.eval_forwarder = EvalForwarder()
        if 'generator_smooth' not in self.models:
            self.models['generator_smooth'] = deepcopy(self.models['generator'])
        load_noise = self.config.test_time_optims == 2
        
        super().load('/media/hdd2/adundar/hamza/genforce/checkpoints/stylegan2_official_ffhq256.pth', 
            running_metadata=False,
                learning_rate=False,
                optimizer=False,
                running_stats=False,
                noise=load_noise,
                load_model = ['discriminator'])

        super().load(self.config.get('gan_model_path'),
                running_metadata=False,
                learning_rate=False,
                optimizer=False,
                running_stats=False,
                noise=load_noise,
                load_model = ['generator_smooth'])
            ## Optimizer for genererator_smooth

    def train_step(self, data, **train_kwargs):
        # if 'generator_smooth' in self.models:
        #     self.set_model_requires_grad('generator_smooth', False)
        # else:
        #     self.set_model_requires_grad('generator', False)
        
        #Train Discriminators
        self.set_model_requires_grad('encoder', False)
        if (self.config.mapping_method != 'pretrained'):
            self.set_model_requires_grad('mapping', False)
        if self.config.create_mixing_network:
            self.set_model_requires_grad('mixer', False)
        # D_latent loss
        if (self.config['use_latent_disc'] == True): 
            self.set_model_requires_grad('latent_disc', True)
            LD_loss, wp_enc, blender = self.loss.ld_loss(self, data) #Forward Encoder once to train both Discriminators
            self.optimizers['latent_disc'].zero_grad()
            LD_loss.backward()
            self.optimizers['latent_disc'].step()
            data['wp_enc'] = wp_enc.detach()
            data['blender'] = None if blender is None else blender.detach()

        # D2_loss
        if (self.config['use_disc2'] == True):
            self.set_model_requires_grad('discriminator2', True)
            D2_loss, wp_enc, blender = self.loss.d2_loss(self, data)
            self.optimizers['discriminator2'].zero_grad()
            D2_loss.backward()
            self.optimizers['discriminator2'].step()
            data['wp_enc'] = wp_enc.detach()
            data['blender'] = None if blender is None else blender.detach()

        # D_loss
        self.set_model_requires_grad('discriminator', True)
        D_loss = self.loss.d_loss(self, data)
        self.optimizers['discriminator'].zero_grad()
        D_loss.backward()
        self.optimizers['discriminator'].step()

        #Train Encoder
        # E_loss
        #self.set_model_requires_grad('encoder', True)
        if (self.config.mapping_method != 'pretrained'):
            self.set_model_requires_grad('mapping', True)
        if self.config.create_mixing_network:
            self.set_model_requires_grad('mixer', True)
        self.set_model_requires_grad('discriminator', False)
        if (self.config['use_latent_disc'] == True):
            self.set_model_requires_grad('latent_disc', False)
        if (self.config['use_disc2'] == True):
            self.set_model_requires_grad('discriminator2', False)
        E_loss = self.loss.e_loss(self, data)
        self.optimizers['generator_smooth'].zero_grad()
        E_loss.backward()
        self.optimizers['generator_smooth'].step()
        # self.optimizers['encoder'].zero_grad()
        # E_loss.backward()
        # self.optimizers['encoder'].step()


    def load(self, **kwargs):
        super().load(**kwargs)

    def train_forward(self, data, iscycle=False):
        return self.train_forwarder.forward(self, data, iscycle)

    def forward_pass(self, data, return_vals='all', only_enc=False):
        return self.eval_forwarder.forward(self, data, only_enc)
    
    def runM(self, z_rand, repeat_w=False):
        G =  self.models['generator_smooth']
        w_rand =  G(z_rand, None, mode='mapping', repeat_w=repeat_w, **self.G_kwargs_val)
        return w_rand
    
    
    def runG(self, data, mode="synthesis", highres_outs=None, return_f = False, resize=True):
        # if 'generator_smooth' in self.models:
        #     G = self.get_module(self.models['generator_smooth'])
        # else:
        #     G = self.get_module(self.models['generator'])
        # G.eval()
        G =  self.models['generator_smooth']

        fakes, gouts =  G(data, highres_outs, return_f=return_f,  **self.G_kwargs_val)
        if return_f:
            return fakes
        if resize:
            fakes = F.adaptive_avg_pool2d(fakes, (256,256))
        return fakes, gouts