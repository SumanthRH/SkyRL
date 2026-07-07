"""Pluggable transports for the delta weight-sync payload (new inference path).

The expensive, model-aware parts of delta sync are transport-independent:

* trainer side -- diff against the bf16 snapshot, pack the sparse ``(positions, values)``
  (:mod:`skyrl.backends.skyrl_train.weight_sync.delta_utils`);
* inference side -- rebuild the NaN-narrow full tensor and merge it onto the per-rank bf16
  shadow via vLLM's own loader (:class:`~...delta_utils.ShardShadow`).

Only the *carrier* of the two packed tensors changes between deployments, so a transport is
the single seam where ``nccl`` and ``disk`` differ:

* :class:`NcclDeltaTransport` -- trainer rank 0 broadcasts the tensors over a PyNccl group
  and every worker broadcast-receives. The validated same-DC baseline.
* :class:`DiskDeltaTransport` -- trainer rank 0 writes one safetensors file per payload batch into a
  versioned directory on a shared filesystem; every worker reads the same file and narrows
  to its TP shard. The cross-DC / low-bandwidth use case.

Both sides agree on the control plane only through the manifest dict (which travels over the
reliable RPC); the bulk payload travels here. This module is intentionally vLLM-free (the
NCCL group object is injected) so it stays importable on CPU for unit tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from loguru import logger

from skyrl.backends.skyrl_train.utils.io import io as skyrl_io
from skyrl.backends.skyrl_train.weight_sync.delta_utils import POSITION_DTYPE


@dataclass
class DeltaPayload:
    """The packed sparse payload for one transport send.

    ``values`` is always present (bf16). ``positions`` (flat indices into the full tensor,
    :data:`POSITION_DTYPE`) is present for a delta and ``None`` for a seed -- a seed is dense,
    so the values *are* the whole flattened tensor and no positions are needed.
    """

    values: torch.Tensor
    positions: Optional[torch.Tensor]

    @property
    def is_seed(self) -> bool:
        return self.positions is None


class DeltaTransport(ABC):
    """Carries one :class:`DeltaPayload` from trainer rank 0 to every worker."""

    # Whether the payload moves over the *same* window the apply RPC is in flight.
    #   * ``True``  (NCCL): the broadcast rendezvous with each worker's receive happens
    #     *during* the apply RPC, so the sender must launch the RPC, then broadcast.
    #   * ``False`` (disk): the file must be durable *before* the apply RPC is issued, so the
    #     sender writes + fsyncs, then issues the RPC.
    sends_during_rpc: bool = True

    # ---- sender side (trainer rank 0) ------------------------------------------------

    def begin_sync(self, version: int) -> None:
        """Hook called once on rank 0 before a sync's chunks (e.g. disk: mkdir version dir)."""

    @abstractmethod
    def send(self, payload: DeltaPayload, *, version: int, file_index: int) -> None:
        """Deliver one payload file/broadcast (no-op when the payload is empty)."""

    def end_sync(self, version: int) -> None:
        """Hook called once on rank 0 after a sync's chunks (e.g. disk: drop old version)."""

    # ---- receiver side (inference worker) --------------------------------------------

    @abstractmethod
    def receive(
        self, *, total: int, is_seed: bool, version: int, file_index: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Obtain one payload file/broadcast.

        Returns ``(values_cpu, positions_cpu)`` -- both on CPU; ``positions_cpu`` is ``None``
        for a seed. ``total`` is the packed element count (``sum(counts)``) and is ``0`` when
        nothing changed, in which case there is nothing on the wire.
        """

    def teardown(self) -> None:
        """Release any transport resources (group handle, open files, ...)."""


class NcclDeltaTransport(DeltaTransport):
    """Broadcast the payload over a vLLM ``PyNcclCommunicator`` group (rank 0 -> all)."""

    sends_during_rpc = True

    def __init__(self, group: Any, device: torch.device) -> None:
        # ``group`` is a PyNcclCommunicator (same object both sides use): the trainer builds
        # it via ``NCCLWeightTransferEngine.trainer_init`` and each worker via the engine's
        # ``init_transfer_engine``. We only call ``.broadcast`` on it, so no vLLM import here.
        self._group = group
        self._device = device

    def _broadcast(self, tensor: torch.Tensor) -> None:
        self._group.broadcast(tensor, src=0, stream=torch.cuda.current_stream())

    def send(self, payload: DeltaPayload, *, version: int, file_index: int) -> None:
        if payload.values.numel() == 0:
            return
        self._broadcast(payload.values.to(self._device))
        if not payload.is_seed:
            assert payload.positions is not None
            self._broadcast(payload.positions.to(self._device))

    def receive(
        self, *, total: int, is_seed: bool, version: int, file_index: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        values = torch.empty(total, dtype=torch.bfloat16, device=self._device)
        if total > 0:
            self._broadcast(values)
        values_cpu = values.cpu()
        if is_seed:
            return values_cpu, None
        positions = torch.empty(total, dtype=POSITION_DTYPE, device=self._device)
        if total > 0:
            self._broadcast(positions)
        return values_cpu, positions.cpu()

    def teardown(self) -> None:
        self._group = None


class DiskDeltaTransport(DeltaTransport):
    """Publish the payload as versioned safetensors objects on a filesystem or object store.

    Layout (one object per batched file)::

        {sync_dir}/weight_v{version:06d}/file_{file_index:05d}.safetensors

    ``sync_dir`` may be a local/shared-FS path *or* a cloud URI (``s3://``, ``gs://``, ...): all
    I/O goes through SkyRL's fsspec helpers (:mod:`skyrl.backends.skyrl_train.utils.io.io`), so
    one code path serves both. The trainer (rank 0) writes; every worker reads the same object
    and narrows to its TP shard via vLLM's loader (positions stay in *full-tensor* index space,
    exactly like the NCCL path). Empty payloads write no object.

    Each sync writes a fresh, never-reused ``weight_v{version}`` path. Object creation is atomic on
    close, and the disk transport issues the apply RPC only *after* the write completes
    (``sends_during_rpc = False``), so no reader ever opens a partial object -- no rename or fsync
    needed. When ``max_files_to_keep`` is set, older file indices in the active version directory
    are removed asynchronously while newer files are uploaded.
    """

    sends_during_rpc = False

    def __init__(self, sync_dir: str, *, max_files_to_keep: Optional[int] = None) -> None:
        if not sync_dir:
            raise ValueError("DiskDeltaTransport requires a non-empty sync_dir")
        if max_files_to_keep is not None and max_files_to_keep < 1:
            raise ValueError("DiskDeltaTransport max_files_to_keep must be >= 1 when set")
        self._sync_dir = sync_dir.rstrip("/")
        self._max_files_to_keep = max_files_to_keep
        self._cleanup_pool: ThreadPoolExecutor | None = None
        self._cleanup_futures: list[Future[None]] = []
        if max_files_to_keep is not None:
            self._cleanup_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="delta-disk-cleanup")

    def _version_dir(self, version: int) -> str:
        return f"{self._sync_dir}/weight_v{version:06d}"

    def _file_path(self, version: int, file_index: int) -> str:
        return f"{self._version_dir(version)}/file_{file_index:05d}.safetensors"

    def _cleanup_path(self, path: str) -> None:
        try:
            if skyrl_io.exists(path):
                skyrl_io.remove(path)
        except Exception:
            logger.opt(exception=True).warning("Failed to cleanup old delta weight file {}", path)

    def _drain_cleanup_futures(self, *, block: bool) -> None:
        remaining: list[Future[None]] = []
        for future in self._cleanup_futures:
            if block or future.done():
                future.result()
            else:
                remaining.append(future)
        self._cleanup_futures = remaining

    def _schedule_cleanup(self, *, version: int, file_index: int) -> None:
        if self._max_files_to_keep is None or self._cleanup_pool is None:
            return
        old_file_index = file_index - self._max_files_to_keep
        if old_file_index < 0:
            return
        self._drain_cleanup_futures(block=False)
        self._cleanup_futures.append(
            self._cleanup_pool.submit(self._cleanup_path, self._file_path(version, old_file_index))
        )

    # ---- sender ----------------------------------------------------------------------

    def begin_sync(self, version: int) -> None:
        # Creates the version dir on a local/shared FS; a no-op for object stores (which have no
        # directories -- the key prefix is implicit in the object path).
        skyrl_io.makedirs(self._version_dir(version), exist_ok=True)

    def send(self, payload: DeltaPayload, *, version: int, file_index: int) -> None:
        self._schedule_cleanup(version=version, file_index=file_index)
        if payload.values.numel() == 0:
            return
        from safetensors.torch import save as st_save

        tensors = {"values": payload.values.detach().cpu().contiguous()}
        metadata = {"is_seed": "1" if payload.is_seed else "0"}
        if not payload.is_seed:
            assert payload.positions is not None
            tensors["positions"] = payload.positions.detach().cpu().contiguous()

        # Serialize to bytes (no local temp file) and write to the unique per-(version, file)
        # path via fsspec -- identical for local FS and cloud object stores.
        data = st_save(tensors, metadata=metadata)
        with skyrl_io.open_file(self._file_path(version, file_index), "wb") as f:
            f.write(data)

    def end_sync(self, version: int) -> None:
        self._drain_cleanup_futures(block=True)

    # ---- receiver --------------------------------------------------------------------

    def receive(
        self, *, total: int, is_seed: bool, version: int, file_index: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if total == 0:
            empty = torch.empty(0, dtype=torch.bfloat16)
            return empty, (None if is_seed else torch.empty(0, dtype=POSITION_DTYPE))
        from safetensors.torch import load as st_load

        with skyrl_io.open_file(self._file_path(version, file_index), "rb") as f:
            loaded = st_load(f.read())
        values_cpu = loaded["values"]
        if is_seed:
            return values_cpu, None
        return values_cpu, loaded["positions"]

    def teardown(self) -> None:
        self._drain_cleanup_futures(block=True)
        if self._cleanup_pool is not None:
            self._cleanup_pool.shutdown(wait=True)
            self._cleanup_pool = None
