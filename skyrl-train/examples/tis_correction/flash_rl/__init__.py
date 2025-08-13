"""
Flash RL source code


All credits to: https://github.com/yaof20/Flash-RL  
"""

from loguru import logger
import os
import sys

logger.add(sys.stdout, level=os.environ.get("FLASHRL_LOGGING_LEVEL", "INFO"))


def check_vllm_installed():
    """Check if vllm is installed"""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def check_dist_initialized():
    """Check if distributed environment is initialized"""
    try:
        from torch.distributed import is_initialized

        return is_initialized()
    except ImportError:
        return False


def apply_patch():
    # Check if patching is needed based on environment variables
    if "FLASHRL_CONFIG" in os.environ and check_vllm_installed():

        from .vllm_patch import patch_vllm_llm, patch_vllm_process_weights_after_loading

        # Patch the process_weights_after_loading function
        process_weights_status = patch_vllm_process_weights_after_loading()
        logger.debug(f"Patching vllm process_weights_after_loading... status: {process_weights_status}")

        # Patch the LLM class
        patch_status = patch_vllm_llm()
        logger.debug(f"Patching the vllm LLM to enable flash_rl quantization... status: {patch_status}")

        if "FLASHRL_TEST_RELOAD" in os.environ:
            from .vllm_patch import patch_vllm_llm_test_reload

            reload_status = patch_vllm_llm_test_reload()
            logger.debug(f"Patching vllm LLM init to test reload... status: {reload_status}")

        if os.environ.get("FLASHRL_LMHEAD_FP32", "0") == "1":
            from .vllm_patch import patch_vllm_lmhead_to_fp32

            patch_status = patch_vllm_lmhead_to_fp32()
            logger.debug(f"Patching vllm lmhead to fp32... status: {patch_status}")
    else:
        logger.debug("Skipping the patching of vllm")
