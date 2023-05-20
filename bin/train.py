import os
from pathlib import Path

import click
from pytorch_lightning.trainer import Trainer

from chunknet.autoencoder import AutoencoderLightning
from chunknet.superchunk import ChunkDataModule, ChunkLoader
from chunknet.util import get_logger, pack_unpack_hooks


logger = get_logger('chunknet.train')


def chunk_loader() -> ChunkLoader:
    return ChunkLoader(Path(os.path.expanduser('~/chunks')))


def calc_num_embeddings() -> int:
    # TODO: Cache this in /tmp or something
    loader = chunk_loader()
    max_emb = 0
    for x in range(loader.dimensions[0]):
        for z in range(loader.dimensions[1]):
            chunk = loader.load_superchunk(x, z)
            max_emb = max(chunk.max(), max_emb)
    return max_emb


def create_model() -> AutoencoderLightning:
    num_embeddings = calc_num_embeddings()
    logger.info(f'num_embeddings={num_embeddings}')
    # TODO: Load from config
    encoder = AutoencoderLightning(
        num_embeddings=num_embeddings,
        embedding_dim=4,
        channels=24,
        multipliers=[1, 2, 4, 4],
        n_resnet_blocks=2,
        z_channels=8,
        dropout=0.1,
        embedding_loss_k=0.01,
        learning_rate=1e-4,
    )
    return encoder


@click.group()
def cli() -> None:
    pass


@cli.command()
def num_embeddings() -> None:
    print(calc_num_embeddings())


@cli.command()
def num_params() -> None:
    n = 0
    size = 0
    model = create_model()
    for param in model.parameters():
        n += param.nelement()
        size += param.nelement() * param.element_size()
    print(f'number of parameters: {n}')
    print(f'size of parameters (MB): {int(size / 1e6)}')


@cli.command()
def train_autoencoder() -> None:
    encoder = create_model()
    # TODO: Separate validation data
    datamodule = ChunkDataModule(chunk_loader(), chunk_loader())
    trainer = Trainer()
    with pack_unpack_hooks():
        trainer.fit(model=encoder, datamodule=datamodule)


if __name__ == '__main__':
    cli()
