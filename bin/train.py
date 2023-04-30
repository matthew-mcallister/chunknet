import logging
import os
from pathlib import Path

import click
from torch import nn

from chunknet.autoencoder import EmbeddingLoss, create_autoencoder
from chunknet.superchunk import ChunkLoader


logger = logging.getLogger('chunknet.train')
logger.setLevel(logging.INFO)


def chunk_loader() -> ChunkLoader:
    return ChunkLoader(Path(os.path.expanduser('~/chunks')))


def calc_num_embeddings() -> int:
    loader = chunk_loader()
    max_emb = 0
    for x in range(loader.dimensions[0]):
        for z in range(loader.dimensions[1]):
            chunk = loader.load_superchunk(x, z)
            max_emb = max(chunk.max(), max_emb)
    return max_emb


@click.group()
def cli() -> None:
    pass


@cli.command()
def num_embeddings() -> None:
    print(calc_num_embeddings())


@cli.command()
def train_autoencoder() -> None:
    logger.info(f'num_embeddings={calc_num_embeddings()}')
    logger.info('num_embeddings', num_embeddings)
    # TOOD: Load from config files
    encoder = create_autoencoder(
        num_embeddings=calc_num_embeddings(),
        embedding_dim=4,
        channels=32,
        multipliers=[1, 2, 4, 4],
        n_resnet_blocks=2,
        z_channels=8,
        dropout=0.1,
    )
    loss_fn = nn.MSELoss()
    embedding_loss = EmbeddingLoss(0.01)


if __name__ == '__main__':
    cli()
