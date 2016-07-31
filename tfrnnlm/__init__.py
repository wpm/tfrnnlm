import logging

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)
