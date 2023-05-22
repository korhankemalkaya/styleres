import argparse
import os
import pickle
import torch
import numpy as np

from editings.styleclip.global_direction import StyleCLIPGlobalDirection
from editings.styleclip.model_ours import Generator


def load_direction_calculator(args):
    delta_i_c = torch.from_numpy(np.load(args['delta_i_c'])).float().cuda()
    with open(args['s_statistics'], "rb") as channels_statistics:
        _, s_std = pickle.load(channels_statistics)
        s_std = [torch.from_numpy(s_i).float().cuda() for s_i in s_std]
    with open(args['text_prompt_templates'], "r") as templates:
        text_prompt_templates = templates.readlines()
    global_direction_calculator = StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates)
    return global_direction_calculator


def load_stylegan_generator(path):
    stylegan_model = Generator(1024, 512, 8, channel_multiplier=2).cuda()
    checkpoint = torch.load(path)
    stylegan_model.load_state_dict(checkpoint['g_ema'])
    return stylegan_model



def styleclip_edit(latent_code_i, conditions, stylegan_model, global_direction_calculator, args):
    #print(f'Editing {image_name}')

    truncation = 1
    mean_latent = None
    input_is_latent = True

    with torch.no_grad():

        _, _, latent_code_s = stylegan_model([latent_code_i],
                                                     efeats = conditions,
                                                     input_is_latent=input_is_latent,
                                                     randomize_noise=False,
                                                     return_latents=True,
                                                     truncation=truncation,
                                                     truncation_latent=mean_latent
                                                    )

    alphas = np.linspace(args['alpha_min'], args['alpha_max'],args['num_alphas'])
    betas = np.linspace(args['beta_min'], args['beta_max'], args['num_betas'])
    for beta in betas:
        direction = global_direction_calculator.get_delta_s(args['neutral_text'], args['target_text'], beta)
        edited_latent_code_s = [[s_i + alpha * b_i for s_i, b_i in zip(latent_code_s, direction)] for alpha in alphas]
        edited_latent_code_s = [torch.cat([edited_latent_code_s[i][j] for i in range(args['num_alphas'])])
                                for j in range(len(edited_latent_code_s[0]))]
        for b in range(0, edited_latent_code_s[0].shape[0]):
            edited_latent_code_s_batch = [s_i[b:b + 1] for s_i in edited_latent_code_s]
            with torch.no_grad():
                edited_image, _, _ = stylegan_model([edited_latent_code_s_batch],
                                                    efeats = conditions,
                                                    input_is_stylespace=True,
                                                    randomize_noise=False,
                                                    return_latents=True,
                                                    )
    return edited_image
