import torch
import torch.nn as nn


def init_net(net, init_type="normal", gpu_ids=[]):
    """Initialize network

    Args:
        net: network
        init_type (str, optional): 'normal', 'xavier', 'kaiming' or 'orthogonal'.
                                    Defaults to "normal".
        gpu_ids (list, optional): GPU IDs. Defaults to [].

    Returns:
        net: initialized network
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.cuda()
        net = torch.nn.DataParallel(net)
    _init_weights(net, init_type)
    return net


def _init_weights(net, init_type="normal", gain=0.02):
    def init_func(layer):
        classname = layer.__class__.__name__
        if hasattr(layer, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(layer.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(layer.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(layer.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(layer.weight.data, 1.0, gain)
            nn.init.constant_(layer.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)
