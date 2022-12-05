import torch
import torch.nn as nn


class EMD_Encoder_Block(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding):
        super(EMD_Encoder_Block, self).__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv2d(
                input_nc,
                output_nc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_nc),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.encoder_block(x)


class EMD_Decoder_Block(nn.Module):
    def __init__(
        self, input_nc, output_nc, kernel_size, stride, padding, add, inner_layer=True
    ):
        super(EMD_Decoder_Block, self).__init__()
        if inner_layer:
            self.decoder_block = nn.Sequential(
                nn.ConvTranspose2d(
                    input_nc,
                    output_nc,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=add,
                    bias=False,
                ),
                nn.BatchNorm2d(output_nc),
                nn.ReLU(inplace=True),
            )
        else:
            self.decoder_block = nn.ConvTranspose2d(
                input_nc,
                output_nc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=add,
                bias=False,
            )

    def forward(self, x):
        return self.decoder_block(x)


class EMD_Content_Encoder(nn.Module):
    def __init__(self, content_channels=1):
        super(EMD_Content_Encoder, self).__init__()
        kernel_sizes = [5, 3, 3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2, 2, 2]
        output_ncs = [64, 128, 256, 512, 512, 512, 512]
        for i in range(7):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            output_nc = output_ncs[i]
            input_nc = output_ncs[i - 1] if i > 0 else content_channels
            padding = kernel_size // 2
            setattr(
                self,
                "encoder_{}".format(i),
                EMD_Encoder_Block(input_nc, output_nc, kernel_size, stride, padding),
            )

    def forward(self, x):
        outps = [x]
        for i in range(7):
            outp = getattr(self, "encoder_{}".format(i))(outps[-1])
            outps.append(outp)
        return outps


class EMD_Style_Encoder(nn.Module):
    def __init__(self, style_channels):
        super(EMD_Style_Encoder, self).__init__()
        kernel_sizes = [5, 3, 3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2, 2, 2]
        output_ncs = [64, 128, 256, 512, 512, 512, 512]
        for i in range(7):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            output_nc = output_ncs[i]
            input_nc = output_ncs[i - 1] if i > 0 else style_channels
            padding = kernel_size // 2
            setattr(
                self,
                "encoder_{}".format(i),
                EMD_Encoder_Block(input_nc, output_nc, kernel_size, stride, padding),
            )

    def forward(self, x):
        for i in range(7):
            x = getattr(self, "encoder_{}".format(i))(x)
        return x


class EMD_Decoder(nn.Module):
    def __init__(
        self,
    ):
        super(EMD_Decoder, self).__init__()
        kernel_sizes = [3, 3, 3, 3, 3, 3, 5]
        strides = [2, 2, 2, 2, 2, 2, 1]
        output_ncs = [512, 512, 512, 256, 128, 64, 1]
        for i in range(7):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            output_nc = output_ncs[i]
            input_nc = output_ncs[i - 1] if i > 0 else 512
            padding = kernel_size // 2
            add = stride // 2
            setattr(
                self,
                "decoder_{}".format(i),
                EMD_Decoder_Block(
                    input_nc * 2,
                    output_nc,
                    kernel_size,
                    stride,
                    padding,
                    add,
                    inner_layer=(i < 6),
                ),
            )
        self.out = nn.Tanh()

    def forward(self, x, layers):
        for i in range(7):
            x = torch.cat([x, layers[-i - 1]], 1)
            x = getattr(self, "decoder_{}".format(i))(x)
        x = self.out(x)
        return x


class EMD_Mixer(nn.Module):
    def __init__(
        self,
    ):
        super(EMD_Mixer, self).__init__()
        self.mixer = nn.Bilinear(512, 512, 512)

    def forward(self, content_feature, style_feature):
        content_feature = torch.squeeze(torch.squeeze(content_feature, -1), -1)
        style_feature = torch.squeeze(torch.squeeze(style_feature, -1), -1)
        mixed = self.mixer(content_feature, style_feature)
        return torch.unsqueeze(torch.unsqueeze(mixed, -1), -1)


class EMD_Generator(nn.Module):
    def __init__(self, style_channels, content_channels=1):
        super(EMD_Generator, self).__init__()
        self.style_encoder = EMD_Style_Encoder(style_channels)
        self.content_encoder = EMD_Content_Encoder(content_channels)
        self.decoder = EMD_Decoder()
        self.mixer = EMD_Mixer()

    def forward(self, style_images, content_images):
        style_feature = self.style_encoder(style_images)
        content_features = self.content_encoder(content_images)
        mixed = self.mixer(content_features[-1], style_feature)
        generated = self.decoder(mixed, content_features)
        return generated
