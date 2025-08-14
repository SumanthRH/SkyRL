"""
Flash RL source code


All credits to: https://github.com/yaof20/Flash-RL  
"""

from loguru import logger
import os
import sys

logger.add(sys.stdout, level=os.environ.get("FLASHRL_LOGGING_LEVEL", "INFO"))


def check_dist_initialized():
    """Check if distributed environment is initialized"""
    try:
        from torch.distributed import is_initialized

        return is_initialized()
    except ImportError:
        return False
