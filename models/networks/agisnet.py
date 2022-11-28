import torch.nn as nn

from .agisblock import AGISNetBlock


class AGISNet(nn.Module):
    def __init__(
        self,
        input_content,
        input_style,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        use_attention=False,
        use_spectral_norm=False,
        upsample="basic",
        wo_skip=False,
    ):
        super(AGISNet, self).__init__()
        max_nchn = 8

        dual_block = AGISNetBlock(
            ngf * max_nchn,
            ngf * max_nchn,
            ngf * max_nchn,
            ngf * max_nchn,
            use_spectral_norm=use_spectral_norm,
            innermost=True,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
            wo_skip=wo_skip,
        )
        for _ in range(num_downs - 5):
            dual_block = AGISNetBlock(
                ngf * max_nchn,
                ngf * max_nchn,
                ngf * max_nchn,
                ngf * max_nchn,
                dual_block,
                norm_layer=norm_layer,
                nl_layer=nl_layer,
                use_dropout=use_dropout,
                use_spectral_norm=use_spectral_norm,
                upsample=upsample,
                wo_skip=wo_skip,
            )
        dual_block = AGISNetBlock(
            ngf * 4,
            ngf * 4,
            ngf * 4,
            ngf * max_nchn,
            dual_block,
            use_attention=use_attention,
            use_spectral_norm=use_spectral_norm,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
            wo_skip=wo_skip,
        )
        dual_block = AGISNetBlock(
            ngf * 2,
            ngf * 2,
            ngf * 2,
            ngf * 4,
            dual_block,
            use_attention=use_attention,
            use_spectral_norm=use_spectral_norm,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
            wo_skip=wo_skip,
        )
        dual_block = AGISNetBlock(
            ngf,
            ngf,
            ngf,
            ngf * 2,
            dual_block,
            use_attention=use_attention,
            use_spectral_norm=use_spectral_norm,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
            wo_skip=wo_skip,
        )
        dual_block = AGISNetBlock(
            input_content,
            input_style,
            output_nc,
            ngf,
            dual_block,
            use_spectral_norm=use_spectral_norm,
            outermost=True,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
            wo_skip=wo_skip,
        )

        self.model = dual_block

    def forward(self, content, style):
        return self.model(content, style)
