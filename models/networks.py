import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.optim.lr_scheduler as lr_scheduler



#--------------------------------INIT NET-------------------------------------

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net = nn.DataParallel(net)
    init_weights(net, init_type)
    return net

#--------------------------------SELF ATTENTION-------------------------------------

# Self Attention module from self-attention gan
class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # print('attention size', x.size())
        m_batchsize, C, width, height = x.size()
        # print('query_conv size', self.query_conv(x).size())
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out

def get_self_attention_layer(in_dim):
    self_attn_layer = SelfAttention(in_dim)
    return self_attn_layer


#--------------------------------NONLINEAR & NORM-------------------------------------

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer


#--------------------------------SPECTRAL NORM-------------------------------------

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero', use_spectral_norm=False):
    # padding_type = 'zero'
    if upsample == 'basic':
        if use_spectral_norm:
            upconv = [SpectralNorm(nn.ConvTranspose2d(
                      inplanes, outplanes, kernel_size=4, stride=2, padding=1))]
        else:
            upconv = [nn.ConvTranspose2d(
                      inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1)]
        if use_spectral_norm:
            upconv += [SpectralNorm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
        else:
            upconv += [nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


#--------------------------------AGIS BLOCK (G BLOCK)------------------------------------

class AGISNetBlock(nn.Module):
    def __init__(self, input_cont, input_style, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, use_spectral_norm=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, use_attention=False,
                 upsample='basic', padding_type='zero', wo_skip=False):
        super(AGISNetBlock, self).__init__()
        self.wo_skip = wo_skip
        p = 0
        downconv1 = []
        downconv2 = []
        if padding_type == 'reflect':
            downconv1 += [nn.ReflectionPad2d(1)]
            downconv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv1 += [nn.ReplicationPad2d(1)]
            downconv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        downconv1 += [nn.Conv2d(input_cont, inner_nc, kernel_size=3, stride=2, padding=p)]
        downconv2 += [nn.Conv2d(input_style, inner_nc, kernel_size=3, stride=2, padding=p)]

        # downsample is different from upsample
        downrelu1 = nn.LeakyReLU(0.2, True)
        downrelu2 = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()
        uprelu2 = nl_layer()

        attn_layer = None
        if use_attention:
            attn_layer = get_self_attention_layer(outer_nc)

        if outermost:
            if self.wo_skip:
                upconv = upsampleLayer(
                    inner_nc, inner_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
            else:
                upconv = upsampleLayer(
                    inner_nc * 4, inner_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc * 3, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)

            upconv_out = [nn.Conv2d(inner_nc + outer_nc, outer_nc, kernel_size=3, stride=1, padding=p)]

            down1 = downconv1
            down2 = downconv2
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B + [nn.Tanh()]

            uprelu3 = nl_layer()
            up_out = [uprelu3] + upconv_out + [nn.Tanh()]
            self.up_out = nn.Sequential(*up_out)

            if use_attention:
                up += [attn_layer]

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]

        elif innermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)
            upconv_B = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type,
                use_spectral_norm=use_spectral_norm)
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
                up_B += [norm_layer(outer_nc)]
        else:
            if self.wo_skip:  # without skip-connection
                upconv = upsampleLayer(
                    inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
            else:
                upconv = upsampleLayer(
                    inner_nc * 4, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
                upconv_B = upsampleLayer(
                    inner_nc * 3, outer_nc, upsample=upsample, padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm)
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            if norm_layer is not None:
                down1 += [norm_layer(inner_nc)]
                down2 += [norm_layer(inner_nc)]
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B

            if use_attention:
                up += [attn_layer]
                attn_layer2 = get_self_attention_layer(outer_nc)
                up_B += [attn_layer2]

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
                up_B += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
                up_B += [nn.Dropout(0.5)]

        self.down1 = nn.Sequential(*down1)
        self.down2 = nn.Sequential(*down2)
        self.submodule = submodule
        self.up = nn.Sequential(*up)
        self.up_B = nn.Sequential(*up_B)

    def forward(self, content, style):

        x1 = self.down1(content)
        x2 = self.down2(style)
        if self.outermost:
            mid_C, mid_B = self.submodule(x1, x2)
            fake_B = self.up_B(mid_B)
            mid_C2 = self.up(mid_C)
            fake_C = self.up_out(torch.cat([mid_C2, fake_B], 1))
            return fake_C, fake_B
        elif self.innermost:
            mid_C = torch.cat([x1, x2], 1)
            mid_B = torch.cat([x1, x2], 1)
            fake_C = self.up(mid_C)
            fake_B = self.up_B(mid_B)
            tmp1 = torch.cat([content, style], 1)
            if self.wo_skip:
                return fake_C, fake_B
            else:
                return torch.cat([torch.cat([fake_C, fake_B], 1), tmp1], 1), torch.cat([fake_B, tmp1], 1)
        else:
            mid, mid_B = self.submodule(x1, x2)
            fake_C = self.up(mid)
            fake_B = self.up_B(mid_B)
            tmp1 = torch.cat([content, style], 1)
            if self.wo_skip:
                return fake_C, fake_B
            else:
                return torch.cat([torch.cat([fake_C, fake_B], 1), tmp1], 1), torch.cat([fake_B, tmp1], 1)

# AGISNet Module
class AGISNet(nn.Module):

    def __init__(self, input_content, input_style, output_nc, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 use_attention=False, use_spectral_norm=False, upsample='basic', wo_skip=False):
        super(AGISNet, self).__init__()
        max_nchn = 8

        dual_block = AGISNetBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, ngf*max_nchn,
                                  use_spectral_norm=use_spectral_norm, innermost=True,
                                  norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        for i in range(num_downs - 5):
            dual_block = AGISNetBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, dual_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                      use_spectral_norm=use_spectral_norm, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(ngf*4, ngf*4, ngf*4, ngf*max_nchn, dual_block, use_attention=use_attention,
                                  use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(ngf*2, ngf*2, ngf*2, ngf*4, dual_block, use_attention=use_attention,
                                  use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(ngf, ngf, ngf, ngf*2, dual_block, use_attention=use_attention,
                                  use_spectral_norm=use_spectral_norm, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        dual_block = AGISNetBlock(input_content, input_style, output_nc, ngf, dual_block,
                                  use_spectral_norm=use_spectral_norm, outermost=True, norm_layer=norm_layer,
                                  nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)

        self.model = dual_block

    def forward(self, content, style):
        return self.model(content, style)

def define_G(input_nc, output_nc, ngf, nencode=4, use_spectral_norm=False,
             norm='batch', nl='relu', use_dropout=False, use_attention=False,
             init_type='xavier', gpu_ids=[], upsample='bilinear'):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    input_content = input_nc
    input_style = input_nc * nencode
    net = AGISNet(input_content, input_style, output_nc, 6, ngf,
                    norm_layer=norm_layer,  nl_layer=nl_layer,
                    use_dropout=use_dropout, use_attention=use_attention,
                    use_spectral_norm=use_spectral_norm, upsample=upsample)

    return init_net(net, init_type, gpu_ids)


#--------------------------------D BLOCK------------------------------------

class D_NLayers(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_spectral_norm=False,
                 norm_layer=None, nl_layer=None, use_sigmoid=False):
        super(D_NLayers, self).__init__()

        kw, padw, use_bias = 4, 1, True
        if use_spectral_norm:
            sequence = [SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                     stride=2, padding=padw, bias=use_bias))]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                  stride=2, padding=padw, bias=use_bias)]
        sequence += [nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if use_spectral_norm:
                sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                          kernel_size=kw, stride=2, padding=padw, bias=use_bias))]
            else:
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)]

        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nl_layer()]

        if use_spectral_norm:
            sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                                      stride=1, padding=0, bias=use_bias))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                                   stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output

def define_D(input_nc, ndf, netD,
             norm='batch', nl='lrelu', use_spectral_norm=False,
             use_sigmoid=False, init_type='xavier', gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_32':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netD == 'basic_64':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer,
                        use_spectral_norm=use_spectral_norm, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, gpu_ids)


#--------------------------------LOSS------------------------------------

class GANLoss(nn.Module):
    def __init__(self, mse_loss=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if mse_loss else nn.BCELoss

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, inputs, target_is_real):
        all_losses = []
        for input in inputs:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss_input = self.loss(input, target_tensor)
            all_losses.append(loss_input)
        loss = sum(all_losses)
        return loss, all_losses

class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCXHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        # featureT: target, featureI: inference
        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX_B = -torch.log(CX)
        CX = torch.mean(CX_B)
        return CX, CX_B

def getScheduler(optimizer, opt):
    def lambda1(epoch): return 1.0 - max(0, epoch-opt.niter) / \
        float(opt.niter_decay + 1)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
