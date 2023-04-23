from dataclasses import dataclass
from pathlib import Path

import numpy as np


SUPERCHUNK_SIZE: int = 14
SUPERCHUNK_SHAPE: tuple[int, int, int] = (
    SUPERCHUNK_SIZE, SUPERCHUNK_SIZE, SUPERCHUNK_SIZE)


class ChunkLoader:
    chunk_dir: Path
    dimensions: tuple[int, int]

    def __init__(self, chunk_dir: Path) -> None:
        self.chunk_dir = chunk_dir

        max_x, max_z = 0, 0
        for p in chunk_dir.glob('*.dat'):
            x, z = p.stem.split(',')
            max_x = max(max_x, int(x))
            max_z = max(max_z, int(z))

        self.dimensions = (max_x, max_z)

    def load_superchunk(self, x: int, z: int) -> np.ndarray:
        path = self.chunk_dir / f'{x},{z}.dat'
        data = path.read_bytes()
        array = np.frombuffer(data, np.short)
        return array.reshape(SUPERCHUNK_SHAPE)

    def load_superchunk_unaligned(self, x: int, z: int) -> np.ndarray:
        result = np.empty(SUPERCHUNK_SHAPE, np.short)
        M = SUPERCHUNK_SIZE

        ch = self.load_superchunk(x // M, z // M)
        result[:M - x % M, :, :M - z % M] = ch[x % M:, :, z % M:]

        if x % M > 0:
            ch = self.load_superchunk(x // M + 1, z // M)
            result[M - x % M:, :, :M - z % M] = ch[:M - x % M, :, z % M:]

        if z % M > 0:
            ch = self.load_superchunk(x // M, z // M + 1)
            result[:M - x % M, :, M - z % M:] = ch[x % M:, :, :M - z % M]

        if x % M > 0 and z % M > 0:
            ch = self.load_superchunk(x // M + 1, z // M + 1)
            result[M - x % M:, :, M - z % M:] = ch[:M - x % M, :, :M - z % M]

        return result
