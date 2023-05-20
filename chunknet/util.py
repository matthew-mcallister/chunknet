from contextlib import contextmanager
import logging
import os
from pathlib import Path
from uuid import UUID
from uuid import uuid4

import torch
from torch import Tensor


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelNamesMapping()[os.getenv('LOG', 'INFO')])
    return logger


def get_tensor_dir() -> Path:
    return Path.home() / '.cache/chunknet/tmp'


def id_to_path(id: UUID) -> Path:
    return get_tensor_dir() / f'{id}.pt'


def pack_hook(x: Tensor) -> UUID:
    print(f'writing tensor of size {x.element_size() * x.nelement() / 1e6:,.2f}mb')
    id = uuid4()
    path = id_to_path(id)
    torch.save(x, path)
    return id


def unpack_hook(id: UUID) -> Tensor:
    print('deleting tensor')
    path = id_to_path(id)
    x = torch.load(path)
    path.unlink()
    return x


@contextmanager
def pack_unpack_hooks():
    get_tensor_dir().mkdir(parents=True, exist_ok=True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield
