import logging
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset


CHUNK_SIZE: int = 16
SUPERCHUNK_SIZE: int = 14 * CHUNK_SIZE
SUPERCHUNK_SHAPE: tuple[int, int, int] = (
    SUPERCHUNK_SIZE, SUPERCHUNK_SIZE, SUPERCHUNK_SIZE)


logger = logging.getLogger('chunknet.train')
logger.setLevel(logging.INFO)


class ChunkLoader(IterableDataset):
    chunk_dir: Path
    dimensions: tuple[int, int]
    sample_radius: int

    def __init__(
        self,
        *,
        chunk_dir: Path,
        sample_radius: int,
    ) -> None:
        super().__init__()

        self.chunk_dir = chunk_dir
        self.sample_radius = sample_radius

        max_x, max_z = 0, 0
        for p in chunk_dir.glob('*.dat'):
            x, z = p.stem.split(',')
            max_x = max(max_x, int(x))
            max_z = max(max_z, int(z))

        self.dimensions = (max_x, max_z)

    def load_superchunk(self, x: int, z: int) -> np.ndarray:
        path = self.chunk_dir / f'{x},{z}.dat'
        logger.debug(f'loading data file: {path}')
        data = path.read_bytes()
        array = np.frombuffer(data, np.int16)
        array = array.astype(np.int32)
        return array.reshape(SUPERCHUNK_SHAPE)

    def load_superchunk_unaligned(self, x: int, z: int) -> np.ndarray:
        result = np.empty(SUPERCHUNK_SHAPE, np.int32)
        M = SUPERCHUNK_SIZE

        ch = self.load_superchunk(x // M, z // M)
        result[:M - x % M, :, :M - z % M] = ch[x % M:, :, z % M:]

        if x % M > 0:
            ch = self.load_superchunk(x // M + 1, z // M)
            result[M - x % M:, :, :M - z % M] = ch[:x % M, :, z % M:]

        if z % M > 0:
            ch = self.load_superchunk(x // M, z // M + 1)
            result[:M - x % M, :, M - z % M:] = ch[x % M:, :, :z % M]

        if x % M > 0 and z % M > 0:
            ch = self.load_superchunk(x // M + 1, z // M + 1)
            result[M - x % M:, :, M - z % M:] = ch[:x % M, :, :z % M]

        return result

    def load_random_super_chunk_unaligned(self) -> np.ndarray:
        x = random.randint(0, SUPERCHUNK_SIZE * self.dimensions[0])
        z = random.randint(0, SUPERCHUNK_SIZE * self.dimensions[1])
        return self.load_superchunk_unaligned(x, z)

    def load_random_cube_unaligned(self, radius: int) -> np.ndarray:
        assert radius < SUPERCHUNK_SIZE
        # XXX: This isn't strictly uniform sampling
        superchunk = self.load_random_super_chunk_unaligned()
        x, y, z = [
            random.randint(0, SUPERCHUNK_SIZE - radius)
            for _ in range(3)
        ]
        return superchunk[x:x + radius, y:y + radius, z:z + radius]

    def steps_per_epoch(self) -> int:
        """Approximates how many steps are required to explore the
        entire dataset.
        """
        num_superchunks = self.dimensions[0] * self.dimensions[1]
        num_samples = num_superchunks * (SUPERCHUNK_SIZE / self.sample_radius)**3
        return int(num_samples)

    def __len__(self) -> int:
        return self.steps_per_epoch()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        return self.load_random_cube_unaligned(self.sample_radius)


class ChunkDataModule(pl.LightningDataModule):
    train_data: ChunkLoader
    validate_data: ChunkLoader

    def __init__(
        self,
        train_data: ChunkLoader,
        validate_data: ChunkLoader,
    ) -> None:
        super().__init__()
        self.train_data = train_data
        self.validate_data = validate_data

    def get_dataloader(self, dataset: ChunkLoader) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=None,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_data)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.validate_data)
