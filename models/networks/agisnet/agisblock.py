import torch
import torch.nn as nn

from .attention import SelfAttention
from .spectralnorm import SpectralNorm


class AGISNetBlock(nn.Module):
    def __init__(
        self,
        input_cont,
        input_style,
        outer_nc,
        inner_nc,
        submodule=None,
        outermost=False,
        innermost=False,
        use_spectral_norm=False,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        use_attention=False,
        upsample="basic",
        padding_type="zero",
        wo_skip=False,
    ):
        super(AGISNetBlock, self).__init__()
        self.wo_skip = wo_skip
        p = 0
        downconv1 = []
        downconv2 = []
        if padding_type == "reflect":
            downconv1 += [nn.ReflectionPad2d(1)]
            downconv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            downconv1 += [nn.ReplicationPad2d(1)]
            downconv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        downconv1 += [
            nn.Conv2d(input_cont, inner_nc, kernel_size=3, stride=2, padding=p)
        ]
        downconv2 += [
            nn.Conv2d(input_style, inner_nc, kernel_size=3, stride=2, padding=p)
        ]

        # downsample is different from upsample
        downrelu1 = nn.LeakyReLU(0.2, True)
        downrelu2 = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()
        uprelu2 = nl_layer()

        attn_layer = None
        if use_attention:
            attn_layer = SelfAttention(outer_nc)

        if outermost:
            if self.wo_skip:
                upconv = get_upsample_layer(
                    inner_nc,
                    inner_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )
                upconv_B = get_upsample_layer(
                    inner_nc,
                    outer_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )
            else:
                upconv = get_upsample_layer(
                    inner_nc * 4,
                    inner_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )
                upconv_B = get_upsample_layer(
                    inner_nc * 3,
                    outer_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )

            upconv_out = [
                nn.Conv2d(
                    inner_nc + outer_nc, outer_nc, kernel_size=3, stride=1, padding=p
                )
            ]

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
            upconv = get_upsample_layer(
                inner_nc * 2,
                outer_nc,
                upsample=upsample,
                padding_type=padding_type,
                use_spectral_norm=use_spectral_norm,
            )
            upconv_B = get_upsample_layer(
                inner_nc * 2,
                outer_nc,
                upsample=upsample,
                padding_type=padding_type,
                use_spectral_norm=use_spectral_norm,
            )
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
                up_B += [norm_layer(outer_nc)]
        else:
            if self.wo_skip:  # without skip-connection
                upconv = get_upsample_layer(
                    inner_nc,
                    outer_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )
                upconv_B = get_upsample_layer(
                    inner_nc,
                    outer_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )
            else:
                upconv = get_upsample_layer(
                    inner_nc * 4,
                    outer_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )
                upconv_B = get_upsample_layer(
                    inner_nc * 3,
                    outer_nc,
                    upsample=upsample,
                    padding_type=padding_type,
                    use_spectral_norm=use_spectral_norm,
                )
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            if norm_layer is not None:
                down1 += [norm_layer(inner_nc)]
                down2 += [norm_layer(inner_nc)]
            up = [uprelu] + upconv
            up_B = [uprelu2] + upconv_B

            if use_attention:
                up += [attn_layer]
                attn_layer2 = SelfAttention(outer_nc)
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
                return torch.cat([torch.cat([fake_C, fake_B], 1), tmp1], 1), torch.cat(
                    [fake_B, tmp1], 1
                )
        else:
            mid, mid_B = self.submodule(x1, x2)
            fake_C = self.up(mid)
            fake_B = self.up_B(mid_B)
            tmp1 = torch.cat([content, style], 1)
            if self.wo_skip:
                return fake_C, fake_B
            else:
                return torch.cat([torch.cat([fake_C, fake_B], 1), tmp1], 1), torch.cat(
                    [fake_B, tmp1], 1
                )


def get_upsample_layer(
    inplanes, outplanes, upsample="basic", padding_type="zeros", use_spectral_norm=False
):
    # padding_type = 'zero'
    if upsample == "basic":
        if use_spectral_norm:
            upconv = [
                SpectralNorm(
                    nn.ConvTranspose2d(
                        inplanes, outplanes, kernel_size=4, stride=2, padding=1
                    )
                )
            ]
        else:
            upconv = [
                nn.ConvTranspose2d(
                    inplanes, outplanes, kernel_size=4, stride=2, padding=1
                )
            ]
    elif upsample == "bilinear":
        upconv = [nn.Upsample(scale_factor=2, mode="bilinear"), nn.ReflectionPad2d(1)]
        if use_spectral_norm:
            upconv += [
                SpectralNorm(
                    nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)
                )
            ]
        else:
            upconv += [
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)
            ]
    else:
        raise NotImplementedError("upsample layer [%s] not implemented" % upsample)
    return upconv
