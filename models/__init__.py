# python3.7
"""Collects all available models together."""

from .model_zoo import MODEL_ZOO
from .stylegan2_official import Generator
from .stylegan2_official_reduced import Generator as GeneratorReduced
from .stylegan2_official import Discriminator
from .psp_inversion import pSpEncoder
from .styletransformer import StyleTransformer
from .e4e_inversion import E4E_Inversion
from .hyperstyle import HyperStyle
from .perceptual_model import PerceptualModel
from .arcface.arcface import ArcFace

__all__ = [
    'MODEL_ZOO', 'Generator', 'Discriminator', 'pSpEncoder', 'StyleTransformer', 'PerceptualModel', 'ArcFace', 'build_generator', 'build_discriminator',
    'build_encoder', 'build_perceptual', 'build_model', 'build_arcface', 'MappingNetwork', 'build_mapping',
    'Encoder4Editing', 'HyperStyle']

_GAN_TYPES_ALLOWED = ['stylegan2_official']
_ENCODER_TYPES_ALLOWED = ['pSp', 'styletransformer', 'e4e', 'hyperstyle']

_MODULES_ALLOWED = ['generator_smooth', 'discriminator2', 'latent_disc', 'generator', 'discriminator', 'encoder', 'perceptual', 'arcface', 'mapping', 'mixer']

def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """

    if gan_type == 'stylegan2_official':
        return Generator(z_dim=512, w_dim=512, c_dim=0, resolution=resolution, img_channels=3, **kwargs)
    if gan_type == 'stylegan2_official_reduced':
        return GeneratorReduced(z_dim=512, w_dim=512, c_dim=0, resolution=resolution, img_channels=3, **kwargs)

    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(gan_type, resolution, **kwargs):
    """Builds discriminator by GAN type.

    Args:
        gan_type: GAN type to which the discriminator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type == 'stylegan2_official' or gan_type == "stylegan2_official_reduced":
        return Discriminator(c_dim=0, img_resolution=resolution, img_channels=3, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_encoder(encoder_type, resolution, **kwargs):
    """Builds encoder by GAN type.

    Args:
        encoder_type: Used encoder type.
        resolution: Input resolution for encoder.
        **kwargs: Additional arguments to build the encoder.

    Raises:
        ValueError: If the `encoder_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if encoder_type not in _ENCODER_TYPES_ALLOWED:
        raise ValueError(f'Invalid Encoder type: `{encoder_type}`!\n'
                         f'Types allowed: {_ENCODER_TYPES_ALLOWED}.')
    if encoder_type == 'pSp':
        return pSpEncoder(resolution, **kwargs)
    if encoder_type == 'styletransformer':
        return StyleTransformer(resolution, **kwargs)
    if encoder_type == 'e4e':
        return E4E_Inversion(resolution, **kwargs)
    if encoder_type == 'hyperstyle':
        return HyperStyle(resolution, **kwargs)

    raise NotImplementedError(f'Unsupported Encoder type `{encoder_type}`!')
def build_perceptual(**kwargs):
    """Builds perceptual model.

    Args:
        **kwargs: Additional arguments to build the encoder.
    """
    return PerceptualModel(**kwargs)
def build_arcface(**kwargs):
    return ArcFace(**kwargs)


def build_model(type, module, resolution=-1, **kwargs):
    """Builds a GAN module (generator/discriminator/etc).

    Args:
        type: Model type to which the model belong.
        module: GAN module to build, such as generator or discrimiantor.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `module` is not supported.
        NotImplementedError: If the `module` is not implemented.
    """
    if module not in _MODULES_ALLOWED:
        raise ValueError(f'Invalid module: `{module}`!\n'
                         f'Modules allowed: {_MODULES_ALLOWED}.')
    if module == 'generator' or module == 'generator_smooth' :
        return build_generator(type, resolution, **kwargs)
    if module == 'discriminator':
        return build_discriminator(type, resolution, **kwargs)
    if module == 'discriminator2':
        return build_discriminator(type, resolution, **kwargs)  #The architectures are same
    if module == 'encoder':
        return build_encoder(type, resolution, **kwargs)
    if module == 'perceptual':
        return build_perceptual(**kwargs)
    if module == 'arcface':
        return build_arcface(**kwargs)
    raise NotImplementedError(f'Unsupported module `{module}`!')
