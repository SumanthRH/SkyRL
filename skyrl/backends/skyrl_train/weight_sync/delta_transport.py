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
* :class:`DiskDeltaTransport` -- trainer rank 0 writes one safetensors file per chunk into a
  versioned directory on a shared filesystem; every worker reads the same file and narrows
  to its TP shard. The cross-DC / low-bandwidth use case.

Both sides agree on the control plane only through the manifest dict (which travels over the
reliable RPC); the bulk payload travels here. This module is intentionally vLLM-free (the
NCCL group object is injected) so it stays importable on CPU for unit tests.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from skyrl.backends.skyrl_train.weight_sync.delta_utils import POSITION_DTYPE


@dataclass
class DeltaPayload:
    """The packed sparse payload for one chunk (concatenated across the chunk's params).

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
    """Carries one chunk's :class:`DeltaPayload` from trainer rank 0 to every worker."""

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
    def send(self, payload: DeltaPayload, *, version: int, chunk_index: int) -> None:
        """Deliver one chunk's payload (no-op when the payload is empty)."""

    def end_sync(self, version: int) -> None:
        """Hook called once on rank 0 after a sync's chunks (e.g. disk: drop old version)."""

    # ---- receiver side (inference worker) --------------------------------------------

    @abstractmethod
    def receive(
        self, *, total: int, is_seed: bool, version: int, chunk_index: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Obtain one chunk's payload.

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

    def send(self, payload: DeltaPayload, *, version: int, chunk_index: int) -> None:
        if payload.values.numel() == 0:
            return
        self._broadcast(payload.values.to(self._device))
        if not payload.is_seed:
            assert payload.positions is not None
            self._broadcast(payload.positions.to(self._device))

    def receive(
        self, *, total: int, is_seed: bool, version: int, chunk_index: int
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
    """Publish the payload as versioned safetensors files on a shared filesystem.

    Layout (one file per chunk, atomically published)::

        {sync_dir}/weight_v{version:06d}/chunk_{chunk_index:05d}.safetensors

    The trainer writes (rank 0 only) and every worker reads the same file, so positions stay
    in *full-tensor* index space exactly like the NCCL path; each worker narrows to its own
    TP shard via vLLM's loader. Empty chunks (no changed elements) write no file.
    """

    sends_during_rpc = False

    def __init__(self, sync_dir: str, keep_files: bool = False) -> None:
        if not sync_dir:
            raise ValueError("DiskDeltaTransport requires a non-empty sync_dir")
        self._sync_dir = sync_dir
        self._keep_files = keep_files

    def _version_dir(self, version: int) -> str:
        return os.path.join(self._sync_dir, f"weight_v{version:06d}")

    def _chunk_path(self, version: int, chunk_index: int) -> str:
        return os.path.join(self._version_dir(version), f"chunk_{chunk_index:05d}.safetensors")

    # ---- sender ----------------------------------------------------------------------

    def begin_sync(self, version: int) -> None:
        os.makedirs(self._version_dir(version), exist_ok=True)

    def send(self, payload: DeltaPayload, *, version: int, chunk_index: int) -> None:
        if payload.values.numel() == 0:
            return
        from safetensors.torch import save_file

        tensors = {"values": payload.values.detach().cpu().contiguous()}
        metadata = {"is_seed": "1" if payload.is_seed else "0"}
        if not payload.is_seed:
            assert payload.positions is not None
            tensors["positions"] = payload.positions.detach().cpu().contiguous()

        path = self._chunk_path(version, chunk_index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        save_file(tensors, tmp_path, metadata=metadata)
        # fsync + atomic rename so a worker on another node never reads a partial file.
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, path)

    def end_sync(self, version: int) -> None:
        if self._keep_files:
            return
        # Keep the current version (just consumed) and drop the previous one: at most two
        # versions live on disk at any time. Older versions were consumed in prior syncs.
        prev_dir = self._version_dir(version - 1)
        if os.path.isdir(prev_dir):
            shutil.rmtree(prev_dir, ignore_errors=True)

    # ---- receiver --------------------------------------------------------------------

    def receive(
        self, *, total: int, is_seed: bool, version: int, chunk_index: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if total == 0:
            empty = torch.empty(0, dtype=torch.bfloat16)
            return empty, (None if is_seed else torch.empty(0, dtype=POSITION_DTYPE))
        from safetensors.torch import load_file

        loaded = load_file(self._chunk_path(version, chunk_index), device="cpu")
        values_cpu = loaded["values"]
        if is_seed:
            return values_cpu, None
        return values_cpu, loaded["positions"]
