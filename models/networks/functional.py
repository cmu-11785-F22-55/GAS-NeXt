import functools

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from .agisnet import AGISNet
from .dnlayer import D_NLayers
from .init import init_net


def define_G(
    input_nc,
    output_nc,
    ngf,
    nencode=4,
    use_spectral_norm=False,
    norm="batch",
    nl="relu",
    use_dropout=False,
    use_attention=False,
    init_type="xavier",
    gpu_ids=[],
    upsample="bilinear",
):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    input_content = input_nc
    input_style = input_nc * nencode
    net = AGISNet(
        input_content,
        input_style,
        output_nc,
        6,
        ngf,
        norm_layer=norm_layer,
        nl_layer=nl_layer,
        use_dropout=use_dropout,
        use_attention=use_attention,
        use_spectral_norm=use_spectral_norm,
        upsample=upsample,
    )

    return init_net(net, init_type, gpu_ids)


def define_D(
    input_nc,
    ndf,
    netD,
    norm="batch",
    nl="lrelu",
    use_spectral_norm=False,
    use_sigmoid=False,
    init_type="xavier",
    gpu_ids=[],
):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = "lrelu"  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == "basic_32":
        net = D_NLayers(
            input_nc,
            ndf,
            n_layers=2,
            norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm,
            nl_layer=nl_layer,
            use_sigmoid=use_sigmoid,
        )
    elif netD == "basic_64":
        net = D_NLayers(
            input_nc,
            ndf,
            n_layers=2,
            norm_layer=norm_layer,
            use_spectral_norm=use_spectral_norm,
            nl_layer=nl_layer,
            use_sigmoid=use_sigmoid,
        )
    else:
        raise NotImplementedError(
            "Discriminator model name [%s] is not recognized" % net
        )
    return init_net(net, init_type, gpu_ids)


def get_non_linearity(layer_type="relu"):
    if layer_type == "relu":
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == "lrelu":
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == "elu":
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            "nonlinearity activitation [%s] is not found" % layer_type
        )
    return nl_layer


def get_norm_layer(layer_type="instance"):
    if layer_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % layer_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    def lambda1(epoch):
        return 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
