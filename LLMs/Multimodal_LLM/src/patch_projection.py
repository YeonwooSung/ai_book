import torch


class PatchProjectionLayer(torch.nn.Module):
    """Implement the patch projection layer using a linear layer."""

    def __init__(self, patch_size, num_channels, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.projection = torch.nn.Linear(
            patch_size * patch_size * num_channels, embedding_dim
        )

    def forward(self, x):
        batch_size, num_patches, channels, height, width = x.size()
        x = x.view(batch_size, num_patches, -1)  # Flatten each patch
        x = self.projection(x)  # Project each flattened patch
        return x


class PatchProjectionConv2DLayer(torch.nn.Module):
        """
        Implement the patch projection layer using a 2D convolutional layer.
        As we could replace linear layer with convolutional layer mathematically,
        we can use a convolutional layer to project the patches.
        """

        def __init__(self, patch_size, num_channels, embedding_dim):
            super().__init__()
            self.patch_size = patch_size
            self.num_channels = num_channels
            self.embedding_dim = embedding_dim
            self.projection = torch.nn.Conv2d(
                num_channels,
                embedding_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size)
            )

        def forward(self, x):
            x = self.projection(x)  # Project each patch
            return x
