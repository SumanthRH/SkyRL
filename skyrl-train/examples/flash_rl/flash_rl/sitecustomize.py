import os
import sys

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)


def check_vllm_installed():
    """Check if vllm is installed"""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def fanout_existing_imports():
    from vllm.model_executor.model_loader import utils

    orig = getattr(utils, "beforeflashrl_process_weights_after_loading", None)
    hacked = utils.process_weights_after_loading
    if not callable(orig) or not callable(hacked):
        return
    for name, mod in list(sys.modules.items()):
        if not name.startswith("vllm.model_executor.model_loader"):
            continue
        for attr_name, attr_value in list(vars(mod).items()):
            if attr_value is orig:
                setattr(mod, attr_name, hacked)
                logger.debug(f"[fanout] Replaced {name}.{attr_name} with hacked fn")


def apply_patch():
    from loguru import logger

    # Check if patching is needed based on environment variables
    if "FLASHRL_CONFIG" in os.environ and check_vllm_installed():

        from examples.flash_rl.flash_rl.vllm_patch import patch_vllm_llm, patch_vllm_process_weights_after_loading

        # Patch the process_weights_after_loading function
        process_weights_status = patch_vllm_process_weights_after_loading()
        fanout_existing_imports()
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


apply_patch()
