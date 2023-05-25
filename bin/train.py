from contextlib import nullcontext
import os
from pathlib import Path

import click
import torch
import yaml
from pytorch_lightning.trainer import Trainer
from typing import Literal

from chunknet.autoencoder import AutoencoderLightning
from chunknet.superchunk import SUPERCHUNK_SIZE, ChunkDataModule, ChunkLoader
from chunknet.util import get_logger, pack_unpack_hooks


logger = get_logger('chunknet.train')


def chunk_loader(sample_radius: int) -> ChunkLoader:
    return ChunkLoader(
        chunk_dir=Path(os.path.expanduser('~/chunks')),
        sample_radius=sample_radius,
    )


def calc_num_embeddings() -> int:
    # TODO: Cache this in /tmp or something
    loader = chunk_loader(SUPERCHUNK_SIZE)
    max_emb = 0
    for x in range(loader.dimensions[0]):
        for z in range(loader.dimensions[1]):
            chunk = loader.load_superchunk(x, z)
            max_emb = max(chunk.max(), max_emb)
    return max_emb + 1


def model_dir() -> Path:
    return Path(__file__).parent.parent / 'models'


def load_model_config(name: str) -> dict:
    path = model_dir() / f'{name}.yaml'
    with path.open() as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> AutoencoderLightning:
    num_embeddings = calc_num_embeddings()
    logger.info(f'num_embeddings={num_embeddings}')
    encoder = AutoencoderLightning(
        num_embeddings=num_embeddings,
        **config,
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
    config = load_model_config('autoencoder')
    model = create_model(config['model'])
    for param in model.parameters():
        n += param.nelement()
        size += param.nelement() * param.element_size()
    print(f'number of parameters: {n}')
    print(f'size of parameters (MB): {int(size / 1e6)}')


def do_train_autoencoder(
    encoder: AutoencoderLightning,
    *,
    sample_radius: int,
) -> None:
    # TODO: Separate validation data
    loader = chunk_loader(sample_radius)
    datamodule = ChunkDataModule(loader, loader)
    trainer = Trainer()
    if os.getenv('PACK'):
        pack = pack_unpack_hooks
    else:
        pack = nullcontext
    with pack():
        trainer.fit(model=encoder, datamodule=datamodule)


@cli.command()
def train_autoencoder() -> None:
    config = load_model_config('autoencoder')
    encoder = create_model(config['model'])
    do_train_autoencoder(encoder, **config['train'])


if __name__ == '__main__':
    cli()
