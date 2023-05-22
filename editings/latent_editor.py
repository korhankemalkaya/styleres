import torch
import sys
from editings import ganspace


class LatentEditor(object):
    def __init__(self, is_cars=False):
        self.is_cars = is_cars  # Since the cars StyleGAN output is 384x512, there is a need to crop the 512x512 output.

    def apply_ganspace(self, latent, ganspace_pca, edit_directions):
        edit_latents = ganspace.edit(latent, ganspace_pca, edit_directions)
        return self._latents_to_image(edit_latents)

    def apply_ganspace_(self, latent, ganspace_pca, edit_directions):
        edit_latents = ganspace.edit(latent, ganspace_pca, edit_directions)
        return edit_latents
        
    def apply_interfacegan(self, latent, direction, factor=1, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent + factor * direction
        return self._latents_to_image(edit_latents)

    def apply_sefa(self, latent, indices=[2, 3, 4, 5], **kwargs):
        edit_latents = sefa.edit(self.generator, latent, indices, **kwargs)
        return self._latents_to_image(edit_latents)

    def apply_sefa_(self, latent, indices=[2, 3, 4, 5], **kwargs):
        edit_latents = sefa.edit(self.generator, latent, indices, **kwargs)
        return edit_latents

    # Currently, in order to apply StyleFlow editings, one should run inference,
    # save the latent codes and load them form the official StyleFlow repository.
    # def apply_styleflow(self):
    #     pass

    def _latents_to_image(self, latents):
        print(latents.shape)
        with torch.no_grad():
            images, _ = self.generator([latents], randomize_noise=False, input_is_latent=True)
            if self.is_cars:
                images = images[:, :, 64:448, :]  # 512x512 -> 384x512
        print("images after generation:", images.shape)
        horizontal_concat_image = torch.cat(list(images), 1)
        print("horizontal images:", images.shape)
        final_image = tensor2im(horizontal_concat_image)
        return final_image
