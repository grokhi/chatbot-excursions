import sys

from loguru import logger as base_logger
from src.core.config import config


def setup_logging():
    base_logger.remove()
    base_logger.add(
        sys.stdout,
        format=(
            "<green>{level}:</green> <cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> - "
            "<level>{message}</level> "
            "<dim>{file}:{line}</dim>"
        ),
        level=config.LOG_LEVEL,
    )
    # if settings.LOG_FILE is not None:
    #     base_logger.add(
    #         settings.LOG_FILE,
    #         rotation="500 MB",
    #         retention="10 days",
    #         level="DEBUG",
    #         backtrace=True,
    #         diagnose=True,
    #     )
    return base_logger


logger = setup_logging()
