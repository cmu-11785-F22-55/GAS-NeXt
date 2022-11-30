from einops.layers.torch import Rearrange
from torch import nn

# helper function


class PatchEmbedding(nn.Module):
    """Segment images into patches"""

    def __init__(self, dim, patch_height, patch_width, channels=3):
        super(PatchEmbedding, self).__init__()
        patch_dim = channels * patch_height * patch_width
        self.layers = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, img):
        return self.layers(img)
