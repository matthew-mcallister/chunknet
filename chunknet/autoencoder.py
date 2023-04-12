from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    embedding: nn.Embedding

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
    ) -> None:
        """Constructs an encoder for the three-dimensional component of
        the latent space.

        First, each block type is represented by an int, and the block
        types are mapped to vectors by an embedding layer. These
        embedded vectors are not to be confused with the final embedding
        of the input data into the latent space.

        The rest of the encoder is modeled closely on the encoder used
        in 2D latent diffusion. Each multiplier defines a block in the
        encoder. After each block, the image is downsampled by a factor
        of two.

        num_embeddings: Number of keys in embedding layer.
        embedding_dim: Dimension of embedded vectors.
        channels: Base number of channels in model.
        multipliers: Defines number and number of channels of
            intermediate blocks. Number of channels is
            `channels * multipliers[i]`.
        n_resnet_blocks: Number of resnet blocks to use in each
            intermediate block.
        z_channels: Number of channels in destination encoding.
        """
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=-1)


class Encoder3d(nn.Module):
    embedding: nn.Embedding
    conv_in: nn.Conv3d
    down: nn.ModuleList
    mid: nn.ModuleList
    norm_out: nn.Module
    conv_out: nn.Conv3d

    def __init__(
        self,
        *,
        embedding: nn.Embedding,
        channels: int,
        multipliers: list[int],
        n_resnet_blocks: int,
        z_channels: int,
    ) -> None:
        super().__init__()

        self.embedding = embedding

        # Initial convolution layer from block embeddings
        self.conv_in = nn.Conv3d(
            embedding.embedding_dim, channels, 3, stride=1, padding=1)

        # Each top-level block consists of a number of ResNet blocks
        # followed by a downsampling layer. Each block may have a
        # different number of channels from the last.
        sizes: list[int] = [m * channels for m in [1] + multipliers]
        self.down = nn.ModuleList()
        for i in range(len(multipliers)):
            self.down.append(CodecBlock(
                in_channels=sizes[i],
                out_channels=sizes[i + 1],
                n_resnet_blocks=n_resnet_blocks,
                # Downsample at the end of each block except the last
                sample=DownSample if i + 1 < len(multipliers) else None,
            ))

        # Final ResNet blocks with attention
        last_size = sizes[-1]
        self.mid = nn.ModuleList((
            ResnetBlock(last_size, last_size),
            AttentionBlock(last_size),
            ResnetBlock(last_size, last_size),
        ))

        # Map to latent space
        self.norm_out = normalization(last_size)
        self.conv_out = nn.Conv3d(last_size, 2 * z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.conv_in(x)
        for block in self.down:
            x = block(x)
        for block in self.mid:
            x = block(x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        *,
        embedding: nn.Embedding,
        channels: int,
        multipliers: list[int],
        n_resnet_blocks: int,
        z_channels: int,
    ) -> None:
        super().__init__()

        sizes: list[int] = [m * channels for m in [1] + multipliers]

        first_size = sizes[-1]
        self.conv_in = nn.Conv3d(first_size, z_channels, 3, padding=1)
        self.mid = nn.ModuleList((
            ResnetBlock(first_size, first_size),
            AttentionBlock(first_size),
            ResnetBlock(first_size, first_size),
        ))

        self.up = nn.ModuleList()
        for i in reversed(range(len(multipliers))):
            self.up.append(CodecBlock(
                in_channels=sizes[i + 1],
                out_channels=sizes[i],
                n_resnet_blocks=n_resnet_blocks,
                sample=UpSample if i > 0 else None,
            ))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(
            channels, embedding.embedding_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.mid:
            x = block(x)
        for block in self.up:
            x = block(x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        # That's it. The output is the embedding vector. You have to
        # pick some lookup method to retrive the original index. Two
        # options: add more layers to train a classifier (gross), or
        # just pick the nearest embedding with respect to some distance
        # measure. As long as the embeddings are properly regularized,
        # there should be no issue with ambiguity.
        return x


class CodecBlock(nn.Module):
    resnet: nn.ModuleList
    sample: nn.Module

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_resnet_blocks: int,
        sample: type[DownSample] | type[UpSample] | None,
    ) -> None:
        # Append resnet blocks
        resnet = nn.ModuleList()
        resnet.append(ResnetBlock(in_channels, out_channels))
        for _ in range(n_resnet_blocks - 1):
            resnet.append(ResnetBlock(out_channels, out_channels))
        self.resnet = resnet

        if sample:
            self.sample = sample(out_channels)
        else:
            self.sample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resnet:
            x = block(x)
        x = self.sample(x)
        return x


class AttentionBlock(nn.Module):
    norm: nn.Module
    q: nn.Module
    k: nn.Module
    v: nn.Module
    proj_out: nn.Module
    scale: float

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Group normalization
        self.norm = normalization(channels)
        # Query, key, and vector matrices
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape `[batch_size, channels, height, width]`
        """
        x_norm = self.norm(x)
        q: torch.Tensor = self.q(x_norm)
        k: torch.Tensor = self.k(x_norm)
        v: torch.Tensor = self.v(x_norm)

        # QKV vectors
        # Reshape to query, key, and vector embeddings from
        # `[batch_size, channels, height, width, depth]` to
        # `[batch_size, channels, height * width * depth]`
        b, c, h, w, d = q.shape
        q = q.view(b, c, h * w * d)
        k = k.view(b, c, h * w * d)
        v = v.view(b, c, h * w * d)

        # softmax(QK^T / sqrt(d_k)) * V
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        out = torch.einsum('bij,bcj->bci', attn, v)

        # Reshape back to `[batch_size, channels, height, width, depth]`
        out = out.view(b, c, h, w, d)
        # Project
        out = self.proj_out(out)
        # Add residual connection
        x = x + out

        return x


class DownSample(nn.Module):
    conv: nn.Module

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1, 0, 1), mode='constant', value=-1)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    conv: nn.Module

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class ResnetBlock(nn.Module):
    norm1: nn.Module
    conv1: nn.Module
    norm2: nn.Module
    conv2: nn.Module
    nin_shortcut: nn.Module

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # First normalization and convolution layer
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        # Second normalization and convolution layer
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = x

        # First normalization and convolution layer
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # Second normalization and convolution layer
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        # Map and add residual
        x = self.nin_shortcut(x) + h
        return x


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


def normalization(channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
