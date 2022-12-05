import torch
import torch.nn as nn
from .resnet import ResnetBlock


class FTGAN_Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(FTGAN_Encoder, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        ]
        for i in range(2):  # add downsampling layers
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        self.model = nn.Sequential(*model)

    def forward(self, inp):
        return self.model(inp)


class FTGAN_Decoder(nn.Module):
    def __init__(self, use_dropout=False, n_blocks=6, ngf=64):
        super(FTGAN_Decoder, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    ngf * 8,
                    padding_type="reflect",
                    norm_layer=nn.BatchNorm2d,
                    use_dropout=use_dropout,
                    use_bias=False,
                )
            ]
        for i in range(2):  # add upsampling layers
            mult = 2 ** (3 - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf * 2, 1, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, inp):
        return self.model(inp)


class FTGAN_Generator_AVG(nn.Module):
    def __init__(self, ngf=64, use_dropout=False, n_blocks=6):
        super(FTGAN_Generator_AVG, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(
            use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf
        )

    def forward(self, inp):
        content_image, style_images = inp
        K = style_images.shape[1]
        content_feature = self.content_encoder(content_image)
        style_features = []
        for i in range(K):
            style_feature = self.style_encoder(
                torch.unsqueeze(style_images[:, i, :, :], dim=1)
            )
            style_features.append(style_feature)
        style_features = torch.mean(torch.stack(style_features, dim=1), dim=1)
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp


class FTGAN_Generator_CAT(nn.Module):
    def __init__(self, style_channels, ngf=64, use_dropout=False, n_blocks=9):
        super(FTGAN_Generator_CAT, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=style_channels, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(
            use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf
        )

    def forward(self, inp):
        content_image, style_images = inp
        content_feature = self.content_encoder(content_image)
        style_features = self.style_encoder(style_images)
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp


class FTGAN_Local_Atten(nn.Module):
    def __init__(self, ngf=64):
        super(FTGAN_Local_Atten, self).__init__()
        self.self_atten = Self_Attn(ngf * 4)
        self.attention = nn.Linear(ngf * 4, 100)
        self.context_vec = nn.Linear(100, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, style_features):
        B, C, H, W = style_features.shape
        h = self.self_atten(style_features)
        h = h.permute(0, 2, 3, 1).reshape(-1, C)
        h = torch.tanh(self.attention(h))  # (B*H*W, 100)
        h = self.context_vec(h)  # (B*H*W, 1)
        attention_map = self.softmax(h.view(B, H * W)).view(B, 1, H, W)  # (B, 1, H, W)
        style_features = torch.sum(style_features * attention_map, dim=[2, 3])
        return style_features


class FTGAN_Global_Atten(nn.Module):
    def __init__(self, ngf=64):
        super(FTGAN_Global_Atten, self).__init__()
        self.ngf = ngf

    def forward(self, style_features, B, K):
        style_features = style_features.view(B, K, self.ngf * 4)
        style_features = torch.mean(style_features, dim=1).view(
            B, self.ngf * 4, 1, 1
        )  # TBD
        style_features = (
            style_features
            + torch.randn([B, self.ngf * 4, 16, 16], device="cuda") * 0.02
        )
        return style_features


class FTGAN_Generator_HAN(nn.Module):
    def __init__(self, ngf=64, use_dropout=False, n_blocks=6):
        super(FTGAN_Generator_HAN, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(
            use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf
        )
        self.local_atten = FTGAN_Local_Atten(ngf=ngf)
        self.global_atten = FTGAN_Global_Atten(ngf=ngf)

    def forward(self, inp):
        content_image, style_images = inp
        B, K, _, _ = style_images.shape
        content_feature = self.content_encoder(content_image)
        style_features = self.style_encoder(style_images.view(-1, 1, 64, 64))
        style_features = self.local_atten(style_features)
        style_features = self.global_atten(style_features, B, K)
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp


class FTGAN_Layer_Atten(nn.Module):
    def __init__(self, ngf=64):
        super(FTGAN_Layer_Atten, self).__init__()
        self.ngf = ngf
        self.fc = nn.Linear(4096, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, style_features, style_features_1, style_features_2, style_features_3, B, K
    ):

        style_features = torch.mean(
            style_features.view(B, K, self.ngf * 4, 4, 4), dim=1
        )
        style_features = style_features.view(B, -1)
        weight = self.softmax(self.fc(style_features))

        style_features_1 = torch.mean(style_features_1.view(B, K, self.ngf * 4), dim=1)
        style_features_2 = torch.mean(style_features_2.view(B, K, self.ngf * 4), dim=1)
        style_features_3 = torch.mean(style_features_3.view(B, K, self.ngf * 4), dim=1)

        style_features = (
            style_features_1 * weight.narrow(1, 0, 1)
            + style_features_2 * weight.narrow(1, 1, 1)
            + style_features_3 * weight.narrow(1, 2, 1)
        ).view(B, self.ngf * 4, 1, 1) + torch.randn(
            [B, self.ngf * 4, 16, 16], device="cuda"
        ) * 0.02
        return style_features


class FTGAN_Generator_MLAN(nn.Module):
    def __init__(self, ngf=64, use_dropout=False, n_blocks=6):
        super(FTGAN_Generator_MLAN, self).__init__()
        self.style_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.content_encoder = FTGAN_Encoder(input_nc=1, ngf=ngf)
        self.decoder = FTGAN_Decoder(
            use_dropout=use_dropout, n_blocks=n_blocks, ngf=ngf
        )
        self.local_atten_1 = FTGAN_Local_Atten(ngf=ngf)
        self.local_atten_2 = FTGAN_Local_Atten(ngf=ngf)
        self.local_atten_3 = FTGAN_Local_Atten(ngf=ngf)
        self.layer_atten = FTGAN_Layer_Atten(ngf=ngf)

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        self.downsample_2 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

    def forward(self, inp):
        content_image, style_images = inp
        B, K, _, _ = style_images.shape
        content_feature = self.content_encoder(content_image)
        style_features = self.style_encoder(style_images.view(-1, 1, 64, 64))
        style_features_1 = self.local_atten_1(style_features)

        style_features = self.downsample_1(style_features)
        style_features_2 = self.local_atten_2(style_features)

        style_features = self.downsample_2(style_features)
        style_features_3 = self.local_atten_3(style_features)

        style_features = self.layer_atten(
            style_features, style_features_1, style_features_2, style_features_3, B, K
        )
        feature = torch.cat([content_feature, style_features], dim=1)
        outp = self.decoder(feature)
        return outp


class Self_Attn(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
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
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
