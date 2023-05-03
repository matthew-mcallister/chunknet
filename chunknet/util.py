import logging
import os


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelNamesMapping()[os.getenv('LOG', 'INFO')])
    return logger
