import torch
import torch.nn as nn
import torch.functional as F


class GANLoss(nn.Module):
    def __init__(self, mse_loss=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
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
        meanT = (
            featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        )
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

        raw_dist = (1.0 - dist) / 2.0

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX_B = -torch.log(CX)
        CX = torch.mean(CX_B)
        return CX, CX_B
