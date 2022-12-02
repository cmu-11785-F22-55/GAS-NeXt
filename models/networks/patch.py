from einops.layers.torch import Rearrange
from torch import nn

# helper function


def to_tuple(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbedding(nn.Module):
    """Segment images into patches"""

    def __init__(
        self,
        dim: int = 512,
        patch_size: int = 8,
        channels: int = 3,
    ) -> None:
        """Intialization of Patch Embedding Module

        Args:
            dim (int, optional): patch embedding dimension. Defaults to 512.
            patch_size (int, optional): size of patch. Defaults to 8.
            channels (int, optional): number of channels of input images. Defaults to 3.
        """
        super(PatchEmbedding, self).__init__()
        patch_height, patch_width = to_tuple(patch_size)
        patch_dim = channels * patch_height * patch_width
        self.layers = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, img):
        return self.layers(img)
