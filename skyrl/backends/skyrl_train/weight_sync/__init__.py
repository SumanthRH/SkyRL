"""Weight synchronization abstractions for distributed RL training."""

from typing import Type

from .base import LoraLoadRequest, WeightChunk, WeightUpdateRequest
from .broadcast_strategy import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferReceiver,
    BroadcastWeightTransferSender,
    BroadcastWeightUpdateRequest,
)
from .cuda_ipc_strategy import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightTransferReceiver,
    CudaIpcWeightTransferSender,
    CudaIpcWeightUpdateRequest,
)
from .delta_strategy import (
    DeltaInitInfo,
    DeltaTransferStrategy,
    DeltaWeightTransferSender,
)
from .delta_utils import (
    POSITION_DTYPE,
    ShardShadow,
    SnapshotDiffer,
    build_full_delta,
    build_full_seed,
    pack_delta,
    pack_seed,
)

# The delta engine subclasses vLLM's NCCLWeightTransferEngine, so importing it requires vLLM.
# Guard the import so this package stays usable in vLLM-less contexts (the engine is only
# needed on inference workers, where it is loaded lazily by vLLM's factory anyway).
try:
    from .delta_engine import (
        DeltaWeightTransferEngine,
        DeltaWeightTransferInitInfo,
        DeltaWeightTransferUpdateInfo,
    )
except ImportError:
    DeltaWeightTransferEngine = None  # type: ignore[assignment,misc]
    DeltaWeightTransferInitInfo = None  # type: ignore[assignment,misc]
    DeltaWeightTransferUpdateInfo = None  # type: ignore[assignment,misc]
from .transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferReceiver,
    WeightTransferSender,
    WeightTransferStrategy,
)
from .weight_extractor import WeightExtractor
from .weight_loader import WeightLoader


def get_transfer_strategy_cls(weight_sync_backend: str, colocate_all: bool) -> Type[WeightTransferStrategy]:
    """Get the appropriate transfer strategy class based on config.

    Selection:
    - "delta": sparse bf16 delta sync (DeltaTransferStrategy). NCCL transport under the hood.
    - "nccl" + colocate_all: CUDA IPC (CudaIpcTransferStrategy).
    - otherwise: broadcast (BroadcastTransferStrategy).

    Args:
        weight_sync_backend: The weight sync backend ("delta", "nccl", or other).
        colocate_all: Whether training and inference are colocated on same nodes.

    Returns:
        The strategy class.
    """
    strategy = get_transfer_strategy(weight_sync_backend, colocate_all)
    if strategy == "ipc":
        return CudaIpcTransferStrategy
    if strategy == "delta":
        return DeltaTransferStrategy
    return BroadcastTransferStrategy


def get_transfer_strategy(weight_sync_backend: str, colocate_all: bool) -> str:
    """Get the appropriate transfer strategy string based on config."""
    if weight_sync_backend == "delta":
        return "delta"
    if weight_sync_backend == "nccl" and colocate_all:
        return "ipc"
    return "nccl"


__all__ = [
    "WeightChunk",
    "WeightExtractor",
    "WeightLoader",
    "WeightUpdateRequest",
    "LoraLoadRequest",
    "BroadcastWeightUpdateRequest",
    "CudaIpcWeightUpdateRequest",
    "WeightTransferStrategy",
    "WeightTransferSender",
    "WeightTransferReceiver",
    "WeightSyncInitInfo",
    "BroadcastInitInfo",
    "CudaIpcInitInfo",
    "BroadcastTransferStrategy",
    "BroadcastWeightTransferSender",
    "BroadcastWeightTransferReceiver",
    "CudaIpcTransferStrategy",
    "CudaIpcWeightTransferSender",
    "CudaIpcWeightTransferReceiver",
    "DeltaInitInfo",
    "DeltaTransferStrategy",
    "DeltaWeightTransferSender",
    "DeltaWeightTransferEngine",
    "DeltaWeightTransferInitInfo",
    "DeltaWeightTransferUpdateInfo",
    "ShardShadow",
    "SnapshotDiffer",
    "POSITION_DTYPE",
    "build_full_seed",
    "build_full_delta",
    "pack_seed",
    "pack_delta",
    "get_transfer_strategy_cls",
    "get_transfer_strategy",
]
