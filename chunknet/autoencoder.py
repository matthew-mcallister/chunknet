from __future__ import annotations

from typing import Any, Callable
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class AutoencoderLightning(pl.LightningModule):
    embedding: nn.Embedding
    encoder_3d: Encoder3d
    decoder_3d: Decoder3d

    learning_rate: float
    loss_fn: nn.MSELoss
    embedding_loss: EmbeddingLoss

    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        channels: int,
        multipliers: list[int],
        n_resnet_blocks: int,
        z_channels: int,
        dropout: float,
        embedding_loss_k: float,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder_3d = Encoder3d(
            embedding=self.embedding,
            channels=channels,
            multipliers=multipliers,
            n_resnet_blocks=n_resnet_blocks,
            z_channels=z_channels,
            dropout=dropout,
        )
        self.decoder_3d = Decoder3d(
            embedding=self.embedding,
            channels=channels,
            multipliers=multipliers,
            n_resnet_blocks=n_resnet_blocks,
            z_channels=z_channels,
            dropout=dropout,
        )
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.embedding_loss = EmbeddingLoss(embedding_loss_k)

    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            # TODO: Configure these?
            betas=(0.5, 0.9),
        )
        return optim

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        embedded: Tensor = self.embedding(batch)
        # Convert from NxNxNxE to ExNxNxN, E = embedding dim
        embedded = torch.permute(embedded, (3, 0, 1, 2))
        # Add trivial batch dimension for places that expect a batch dim,
        # namely group norm
        embedded = embedded.reshape((1, *embedded.shape))
        round_trip = self.decoder_3d(self.encoder_3d(embedded))

        loss = self.loss_fn(round_trip, embedded)
        self.log('loss', loss)
        emb_loss = self.embedding_loss(self.embedding.weight)
        self.log('emb_loss', emb_loss)
        return loss + emb_loss


class Encoder3d(nn.Module):
    # TODO: Quantization?
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
        dropout: float,
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
                dropout=dropout,
            ))

        # Final ResNet blocks with attention
        last_size = sizes[-1]
        self.mid = nn.ModuleList((
            ResnetBlock(last_size, last_size, dropout=dropout),
            AttentionBlock(last_size),
            ResnetBlock(last_size, last_size, dropout=dropout),
        ))

        # Map to latent space
        self.norm_out = normalization(last_size)
        self.conv_out = nn.Conv3d(last_size, 2 * z_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
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
        dropout: float,
    ) -> None:
        super().__init__()

        sizes: list[int] = [m * channels for m in [1] + multipliers]

        first_size = sizes[-1]
        self.conv_in = nn.Conv3d(first_size, z_channels, 3, padding=1)
        self.mid = nn.ModuleList((
            ResnetBlock(first_size, first_size, dropout=dropout),
            AttentionBlock(first_size),
            ResnetBlock(first_size, first_size, dropout=dropout),
        ))

        self.up = nn.ModuleList()
        for i in reversed(range(len(multipliers))):
            self.up.append(CodecBlock(
                in_channels=sizes[i + 1],
                out_channels=sizes[i],
                n_resnet_blocks=n_resnet_blocks,
                sample=UpSample if i > 0 else None,
                dropout=dropout,
            ))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(
            channels, embedding.embedding_dim, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.mid:
            x = block(x)
        for block in self.up:
            x = block(x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

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
        dropout: float,
    ) -> None:
        super().__init__()

        # Append resnet blocks
        resnet = nn.ModuleList()
        resnet.append(ResnetBlock(in_channels, out_channels, dropout=dropout))
        for _ in range(n_resnet_blocks - 1):
            resnet.append(ResnetBlock(
                out_channels, out_channels, dropout=dropout))
        self.resnet = resnet

        if sample:
            self.sample = sample(out_channels)
        else:
            self.sample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        """
        x: tensor of shape `[batch_size, channels, height, width]`
        """
        x_norm = self.norm(x)
        q: Tensor = self.q(x_norm)
        k: Tensor = self.k(x_norm)
        v: Tensor = self.v(x_norm)

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

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 1, 0, 1, 0, 1), mode='constant', value=-1)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    conv: nn.Module

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class ResnetBlock(nn.Module):
    norm1: nn.Module
    conv1: nn.Module
    norm2: nn.Module
    dropout: nn.Module
    conv2: nn.Module
    nin_shortcut: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # First normalization and convolution layer
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        # Second normalization and convolution layer
        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = x

        # First normalization and convolution layer
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # Second normalization and convolution layer
        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Map and add residual
        x = self.nin_shortcut(x) + h
        return x


def swish(x: Tensor):
    return x * torch.sigmoid(x)


def normalization(channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)


class EmbeddingLoss(nn.Module):
    """Regularization term to prevent degenerate embeddings from arising.

    Because the training data is embedded before running the model, a
    constraint is needed to ensure the embedding is valid. Namely,
    embedding vectors must not be too close to each other, as that
    would prevent reconstruction of the output from the embedding
    vectors.
    """
    k: float

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = k

    def forward(self, embedding_weight: Tensor) -> Tensor:
        # TODO?: This is O(n^2) in number of embeddings and can probably
        # be made more efficient.
        num_embeddings = embedding_weight.shape[0]
        r = torch.cat([
            embedding_weight[:num_embeddings - i] - embedding_weight[i:]
            for i in range(1, num_embeddings)
        ])
        r2 = torch.sum(r * r, dim=1)
        x2 = r2 / self.k**2
        loss = torch.exp(-x2) / x2
        return loss.sum()
