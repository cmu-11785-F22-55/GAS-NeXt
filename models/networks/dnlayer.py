import torch.nn as nn
from .spectralnorm import SpectralNorm


class D_NLayers(nn.Module):
    def __init__(
        self,
        input_nc=3,
        ndf=64,
        n_layers=3,
        use_spectral_norm=False,
        norm_layer=None,
        nl_layer=None,
        use_sigmoid=False,
    ):
        super(D_NLayers, self).__init__()

        kw, padw, use_bias = 4, 1, True
        if use_spectral_norm:
            sequence = [
                SpectralNorm(
                    nn.Conv2d(
                        input_nc,
                        ndf,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    )
                )
            ]
        else:
            sequence = [
                nn.Conv2d(
                    input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias
                )
            ]
        sequence += [nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if use_spectral_norm:
                sequence += [
                    SpectralNorm(
                        nn.Conv2d(
                            ndf * nf_mult_prev,
                            ndf * nf_mult,
                            kernel_size=kw,
                            stride=2,
                            padding=padw,
                            bias=use_bias,
                        )
                    )
                ]
            else:
                sequence += [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    )
                ]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            )
        ]

        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nl_layer()]

        if use_spectral_norm:
            sequence += [
                SpectralNorm(
                    nn.Conv2d(
                        ndf * nf_mult,
                        1,
                        kernel_size=kw,
                        stride=1,
                        padding=0,
                        bias=use_bias,
                    )
                )
            ]
        else:
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=0, bias=use_bias
                )
            ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output
