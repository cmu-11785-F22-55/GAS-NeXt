import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from torch.optim import lr_scheduler

from .vgg import VGG19


def define_G(input_nc, output_nc, nz, ngf, nencode=4, netG='unet_128', use_spectral_norm=False,
             norm='batch', nl='relu', use_dropout=False, use_attention=False,
             init_type='xavier', gpu_ids=[], where_add='input', upsample='bilinear'):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if nz == 0:
        where_add = 'input'

    if netG == 'agisnet':
        input_content = input_nc
        input_style = input_nc * nencode
        net = AGISNet(input_content, input_style, output_nc, 6, ngf,
                      norm_layer=norm_layer,  nl_layer=nl_layer,
                      use_dropout=use_dropout, use_attention=use_attention,
                      use_spectral_norm=use_spectral_norm, upsample=upsample)

    elif netG == 'unet_64' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 6, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_128' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, use_attention=use_attention,
                               use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_64' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 6, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, use_attention=use_attention,
                             use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_128' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, use_attention=use_attention,
                             use_spectral_norm=use_spectral_norm, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, use_attention=use_attention,
                             use_spectral_norm=use_spectral_norm, upsample=upsample)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)
