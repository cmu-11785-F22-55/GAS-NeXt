import torch
import torch.nn as nn


def conv(in_c, out_c, act=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        act(),
    )


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.feature_0 = conv(3, 64)
        self.feature_2 = conv(64, 64)
        self.feature_5 = conv(64, 128)
        self.feature_7 = conv(128, 128)
        self.feature_10 = conv(128, 256)
        self.feature_12 = conv(256, 256)
        self.feature_14 = conv(256, 256)
        self.feature_16 = conv(256, 256)
        self.feature_19 = conv(256, 512)
        self.feature_21 = conv(512, 512)

    def load_model(self, path):
        checkpoint = torch.load(path)
        my_vgg = self.state_dict()
        for my_key, checkpoint_val in zip(my_vgg.keys(), checkpoint.values()):
            my_vgg[my_key] = checkpoint_val
        self.load_state_dict(my_vgg)

    def forward(self, x):
        dict = {}
        x = self.feature_0(x)
        x = self.feature_2(x)
        x = self.avgpool(x)
        x = self.feature_5(x)
        x = self.feature_7(x)
        x = self.avgpool(x)
        x = self.feature_10(x)
        x = self.feature_12(x)
        x = self.feature_14(x)
        dict["conv3_3"] = x
        x = self.feature_16(x)
        x = self.maxpool(x)
        x = self.feature_19(x)
        x = self.feature_21(x)
        dict["conv4_2"] = x
        return dict
