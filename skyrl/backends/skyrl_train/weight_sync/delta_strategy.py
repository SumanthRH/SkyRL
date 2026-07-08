"""Delta weight transfer strategy (sparse bf16 deltas).

This strategy ships only the *changed* bf16 elements each sync instead of full tensors,
which makes weight sync cheap on bandwidth-bound (e.g. cross-DC) setups. The diff/encode
(sender) and NaN-narrow decode + shadow merge (receiver) are transport-independent; the
packed ``(positions, values)`` payload is carried by a pluggable
:class:`~skyrl.backends.skyrl_train.weight_sync.delta_transport.DeltaTransport` -- ``"nccl"``
(broadcast, the same group mechanism as :class:`BroadcastTransferStrategy`) or ``"disk"``
(versioned safetensors on a shared filesystem), selected via
``InferenceEngineConfig.delta_weight_sync_config``. Disk transport is supported only on the
new-inference path.

* **Sender** (trainer rank 0) keeps a bf16 snapshot of what it last sent, diffs the freshly
  extracted weights against it, and ships only ``(positions, values)`` for the changed
  elements (the first sync per parameter is a dense *seed*).
* **Receiver** is vLLM's ``"delta"`` ``WeightTransferEngine``
  (:class:`~skyrl.backends.skyrl_train.weight_sync.delta_engine.DeltaWeightTransferEngine`),
  driven via ``start_weight_update`` / ``update_weights_nccl`` / ``finish_weight_update`` and
  fed by :meth:`DeltaWeightTransferSender._send_chunks_vllm_native`. It keeps a **pinned-CPU
  bf16 shard shadow** (:class:`~...delta_utils.ShardShadow`) -- the high-precision master of
  *this worker's local shard* -- and, for each touched parameter, reconstructs a fresh
  **full** CPU tensor that is ``NaN`` everywhere except the changed positions (dense for a
  seed). vLLM's load path narrows full->shard, the shadow's patched ``copy_`` NaN-masks so
  only changed shard positions overwrite the previous shard, and re-quantizes -- so this
  works for quantized inference too.

See ``vllm/ai_notes/05_delta_loadweights_cpu_shadow_review.md`` for the design.

Notes / v1 scope:
* **New-inference path only** (``_SKYRL_USE_NEW_INFERENCE``); the legacy SkyRL-receiver path
  is not supported for delta sync.
* Requires a **bf16** transfer dtype (the shadow master is bf16).
* The wire is sparse, but every parameter is still yielded each sync, so re-quant compute is
  not yet reduced (only bandwidth). Touched-layer-only re-quant is a documented follow-up.
"""

import asyncio
import math
import socket
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import ray
import torch
from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.delta_transport import (
    DeltaPayload,
    DeltaTransport,
    DiskDeltaTransport,
    NcclDeltaTransport,
)
from skyrl.backends.skyrl_train.weight_sync.delta_utils import (
    POSITION_DTYPE,
    SnapshotDiffer,
    build_delta_manifest,
    delta_checksum,
    pack_delta,
    pack_seed,
    position_dtype_from_name,
    position_dtype_name,
    validate_position_dtype_for_shapes,
)
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferReceiver,
    WeightTransferSender,
    WeightTransferStrategy,
)
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE

if TYPE_CHECKING:
    from skyrl.train.config.config import InferenceEngineConfig


def _format_delta_size(
    is_seed: bool,
    counts: List[int],
    shapes: Iterable[Iterable[int]],
    packed_values: torch.Tensor,
    packed_positions: Optional[torch.Tensor],
) -> str:
    """One-line summary of a delta chunk: fraction of weights changed and wire size."""
    nnz = sum(counts)
    total_elems = sum(int(math.prod(s)) for s in shapes)
    values_bytes = packed_values.numel() * packed_values.element_size()
    positions_bytes = (
        packed_positions.numel() * packed_positions.element_size()
        if (not is_seed and packed_positions is not None)
        else 0
    )
    delta_bytes = values_bytes + positions_bytes
    pct = (100.0 * nnz / total_elems) if total_elems else 0.0
    return (
        f"delta chunk: {'seed' if is_seed else 'sparse'} params={len(counts)} "
        f"nnz={nnz}/{total_elems} ({pct:.2f}% of weights) "
        f"size={delta_bytes / 2**20:.2f} MiB ({delta_bytes} B; "
        f"values={values_bytes} B, positions={positions_bytes} B)"
    )


def _payload_wire_bytes(payload: DeltaPayload) -> int:
    bytes_ = payload.values.numel() * payload.values.element_size()
    if not payload.is_seed and payload.positions is not None:
        bytes_ += payload.positions.numel() * payload.positions.element_size()
    return bytes_


@dataclass
class _PreparedDeltaChunk:
    names: List[str]
    shapes: List[List[int]]
    counts: List[int]
    is_seed: bool
    payload: DeltaPayload
    wire_bytes: int


@dataclass
class _PendingDeltaFile:
    is_seed: bool
    names: List[str]
    dtype_names: List[str]
    shapes: List[List[int]]
    counts: List[int]
    values: List[torch.Tensor]
    positions: List[torch.Tensor]
    positions_dtype: Optional[torch.dtype] = None
    wire_bytes: int = 0

    @classmethod
    def empty(cls, *, is_seed: bool) -> "_PendingDeltaFile":
        return cls(
            is_seed=is_seed,
            names=[],
            dtype_names=[],
            shapes=[],
            counts=[],
            values=[],
            positions=[],
        )

    def add(self, prepared: _PreparedDeltaChunk, dtype_str: str) -> None:
        if prepared.is_seed != self.is_seed:
            raise ValueError("Cannot mix seed and sparse delta payloads in one delta file")
        self.names.extend(prepared.names)
        self.dtype_names.extend([dtype_str] * len(prepared.names))
        self.shapes.extend(prepared.shapes)
        self.counts.extend(prepared.counts)
        self.values.append(prepared.payload.values)
        if not self.is_seed:
            assert prepared.payload.positions is not None
            positions_dtype = prepared.payload.positions.dtype
            if self.positions_dtype is None:
                self.positions_dtype = positions_dtype
            elif self.positions_dtype != positions_dtype:
                raise ValueError(
                    f"Cannot mix delta position dtypes in one file: "
                    f"{self.positions_dtype} and {positions_dtype}"
                )
            self.positions.append(prepared.payload.positions)
        self.wire_bytes += prepared.wire_bytes

    def to_payload(self) -> DeltaPayload:
        if self.values:
            values = torch.cat([v.detach().to(torch.bfloat16).reshape(-1) for v in self.values])
        else:
            values = torch.empty(0, dtype=torch.bfloat16)
        if self.is_seed:
            return DeltaPayload(values=values, positions=None)
        positions_dtype = self.positions_dtype or POSITION_DTYPE
        if self.positions:
            positions = torch.cat([p.detach().to(positions_dtype).reshape(-1) for p in self.positions])
        else:
            positions = torch.empty(0, dtype=positions_dtype)
        return DeltaPayload(values=values, positions=positions)


@dataclass
class DeltaInitInfo(WeightSyncInitInfo):
    """Initialization info for delta weight transfer.

    The NCCL group params (``master_addr``/``master_port``/``world_size``/``backend``) are
    always populated but only used when ``transport == "nccl"``; the disk transport ignores
    them and uses ``sync_dir`` instead.
    """

    master_addr: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str
    backend: str
    model_dtype_str: str
    transport: str = "nccl"
    """Payload carrier: ``"nccl"`` (broadcast) or ``"disk"`` (shared-FS safetensors)."""
    sync_dir: Optional[str] = None
    """Local/shared-FS path or cloud URI (``s3://``/``gs://``) for ``transport="disk"`` (required
    for disk; ignored for NCCL)."""
    max_file_size_in_gb: float = 1.0
    """Maximum batched delta file size in GiB for ``transport="disk"``."""
    max_files_to_keep: Optional[int] = None
    """Optional per-sync retention limit for disk delta files."""
    positions_transfer_dtype: str = "int32"
    """Integer dtype used to transfer sparse flat positions: ``"int32"`` or ``"int64"``."""
    trainer_diff_stage_area: str = "cpu"
    """Where trainer rank 0 computes deltas: ``"cpu"`` or ``"gpu"``."""
    diff_num_workers: int = 0
    """Maximum per-chunk diff workers. ``0`` means one worker per tensor."""

    @staticmethod
    def strategy_type() -> type:
        return DeltaTransferStrategy

    def for_engine(self, engine_index: int, tp_size: int, pp_size: int, dp_size: int) -> "DeltaInitInfo":
        cumulative_offset = engine_index * tp_size * pp_size * dp_size
        return replace(self, rank_offset=self.rank_offset + cumulative_offset)

    def for_servers(self, world_size_per_server: int, num_servers: int, dp_size: int = 1) -> List["DeltaInitInfo"]:
        """Return one DeltaInitInfo per server with the cumulative rank_offset (new path).

        Mirrors ``BroadcastInitInfo.for_servers``: the new-inference path expands the single
        init_info into a list (one per server), then posts ``to_api_payload()`` per server to
        ``/init_weight_transfer_engine``. DP servers within one deployment share a rank_offset
        because vLLM's ``init_transfer_engine`` already accounts for ``dp_rank`` internally; the
        offset only advances at deployment (num_engines) boundaries.
        """
        result: List["DeltaInitInfo"] = []
        rank_offset = self.rank_offset
        for i in range(num_servers):
            result.append(replace(self, rank_offset=rank_offset))
            if (i + 1) % dp_size == 0:
                rank_offset += world_size_per_server
        return result

    def to_api_payload(self) -> Dict[str, Any]:
        """JSON payload for the ``/init_weight_transfer_engine`` endpoint (new path).

        Maps to ``DeltaWeightTransferInitInfo`` fields; ``model_dtype_name`` tells the engine
        the (bf16) transfer dtype and ``transport``/``sync_dir`` select the payload carrier the
        engine builds in ``init_transfer_engine``.
        """
        return {
            "master_address": self.master_addr,
            "master_port": self.master_port,
            "rank_offset": self.rank_offset,
            "world_size": self.world_size,
            "model_dtype_name": self.model_dtype_str,
            "transport": self.transport,
            "sync_dir": self.sync_dir or "",
        }


class DeltaWeightTransferSender(WeightTransferSender):
    """Sends sparse bf16 deltas (trainer side) via a pluggable :class:`DeltaTransport`."""

    def __init__(
        self,
        init_info: DeltaInitInfo,
        inference_client: InferenceEngineClient,
        transport: Optional[DeltaTransport] = None,
    ) -> None:
        self._init_info = init_info
        # Payload carrier on rank 0 (NCCL group or shared-FS disk); ``None`` on other ranks.
        self._transport = transport
        self._inference_client = inference_client
        # Snapshot lives only on the rank that diffs/sends (rank 0).
        self._differ = SnapshotDiffer()
        # Monotonic per-sync version, used to route disk writes/reads to a versioned dir.
        self._version = -1
        # Tracked on every rank (not just rank 0) so the disk-seed skip decision stays in
        # lockstep across ranks. Flips True after the first ``send_chunks``.
        self._snapshot_seeded = False

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        await self._send_chunks_vllm_native(chunks)

    async def _send_chunks_vllm_native(self, chunks: Iterable[WeightChunk]) -> None:
        """New-inference path: drive vLLM's ``delta`` engine via start/update/finish.

        Rank 0 wraps the whole sync in ``start_weight_update`` ... ``finish_weight_update`` and,
        per chunk, posts the sparse manifest (``update_weights_nccl``) while the transport
        delivers the packed ``values`` (and ``positions`` for a delta). All ranks iterate the
        chunks (weight extraction may use collective ops) and barrier in lockstep. The transport
        (NCCL broadcast vs shared-FS disk) is the only thing that differs by deployment.

        Exception: the **first** sync over the **disk** transport ships nothing -- it only
        records the trainer-side snapshot, since the engine already holds the base weights (see
        the inline note below).
        """
        rank = torch.distributed.get_rank()

        # Disk seed: the first sync over disk ships *nothing* -- it only records the trainer-
        # side snapshot so the next sync produces a correct sparse diff. The inference engine
        # already holds the base weights: its shadow is primed at init with the true bf16
        # master (a live snapshot for a bf16 engine, or the load-time pre-quant capture -- see
        # ``capture_layer_prequant_bf16`` -- for a quantized engine), so writing the full model
        # to the shared FS would be redundant. (NCCL still ships the seed: it is the validated
        # losslessness baseline.) Gated on the config (which every rank has) rather than the
        # transport object (only rank 0 holds it) so the skip decision is identical on all ranks
        # and the barrier dance stays in lockstep.
        _DISK_SKIP_SEED = True
        if _DISK_SKIP_SEED and self._init_info.transport == "disk" and not self._snapshot_seeded:
            for chunk in chunks:
                if rank == 0:
                    self._differ.seed_many(
                        chunk.names,
                        chunk.tensors,
                        num_workers=self._diff_num_workers_for_chunk(chunk),
                    )
                torch.distributed.barrier()
            self._snapshot_seeded = True
            return

        if rank == 0:
            assert self._transport is not None, "Rank 0 must have a delta transport"
            self._version += 1
            await self._inference_client.start_weight_update(is_checkpoint_format=True)
            self._transport.begin_sync(self._version)

        if self._init_info.transport == "disk":
            await self._send_chunks_disk(chunks, rank)
        else:
            file_index = 0
            for chunk in chunks:
                if rank == 0:
                    await self._send_one_chunk_native(chunk, self._version, file_index)
                    file_index += 1
                torch.distributed.barrier()

        if rank == 0:
            await self._inference_client.finish_weight_update()
            self._transport.end_sync(self._version)
        torch.distributed.barrier()
        self._snapshot_seeded = True

    def _prepare_delta_chunk(self, chunk: WeightChunk) -> _PreparedDeltaChunk:
        # Seed the whole chunk if any parameter hasn't been sent before; otherwise delta.
        is_seed = not all(self._differ.has(n) for n in chunk.names)
        diff_stage_area = self._init_info.trainer_diff_stage_area
        diff_num_workers = self._diff_num_workers_for_chunk(chunk)

        if is_seed:
            values = self._differ.seed_many(chunk.names, chunk.tensors, num_workers=diff_num_workers)
            packed_values, counts = pack_seed(values)
            packed_positions = None
        else:
            position_dtype = position_dtype_from_name(self._init_info.positions_transfer_dtype)
            validate_position_dtype_for_shapes(chunk.names, chunk.shapes, position_dtype)
            diffs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for did_seed, pos, val in self._differ.diff_many(
                chunk.names,
                chunk.tensors,
                backend=diff_stage_area,
                num_workers=diff_num_workers,
            ):
                if did_seed:
                    raise ValueError("Delta diff unexpectedly reseeded a parameter in a sparse chunk.")
                assert pos is not None
                diffs.append((pos, val))
            packed_positions, packed_values, counts = pack_delta(diffs, position_dtype=position_dtype)

        logger.info(_format_delta_size(is_seed, counts, chunk.shapes, packed_values, packed_positions))

        payload = DeltaPayload(values=packed_values, positions=None if is_seed else packed_positions)
        return _PreparedDeltaChunk(
            names=list(chunk.names),
            shapes=[list(s) for s in chunk.shapes],
            counts=counts,
            is_seed=is_seed,
            payload=payload,
            wire_bytes=_payload_wire_bytes(payload),
        )

    def _diff_num_workers_for_chunk(self, chunk: WeightChunk) -> int:
        if self._init_info.diff_num_workers > 0:
            return min(self._init_info.diff_num_workers, len(chunk.names))
        return len(chunk.names)

    async def _send_chunks_disk(self, chunks: Iterable[WeightChunk], rank: int) -> None:
        dtype_str = self._init_info.model_dtype_str
        max_file_bytes = max(1, int(self._init_info.max_file_size_in_gb * (2**30)))
        pending: _PendingDeltaFile | None = None
        file_index = 0

        async def flush_pending() -> None:
            nonlocal pending, file_index
            if pending is None:
                return
            if pending.wire_bytes == 0:
                pending = None
                return
            payload = pending.to_payload()
            checksum = delta_checksum(None if pending.is_seed else payload.positions, payload.values)
            if pending.is_seed:
                positions_dtype = self._init_info.positions_transfer_dtype
            else:
                assert payload.positions is not None
                positions_dtype = position_dtype_name(payload.positions.dtype)
            update_info = build_delta_manifest(
                names=pending.names,
                dtype_names=pending.dtype_names,
                shapes=pending.shapes,
                counts=pending.counts,
                is_seed=pending.is_seed,
                checksum=checksum,
                version=self._version,
                file_index=file_index,
                positions_dtype=positions_dtype,
            )
            logger.info(
                "delta file: version={} file_index={} params={} size={:.2f} MiB",
                self._version,
                file_index,
                len(pending.names),
                pending.wire_bytes / 2**20,
            )
            assert self._transport is not None
            await asyncio.to_thread(self._transport.send, payload, version=self._version, file_index=file_index)
            await self._inference_client.update_weights_nccl(update_info)
            file_index += 1
            pending = None

        for chunk in chunks:
            if rank == 0:
                prepared = self._prepare_delta_chunk(chunk)
                if pending is not None and pending.is_seed != prepared.is_seed:
                    await flush_pending()
                if (
                    pending is not None
                    and pending.wire_bytes > 0
                    and prepared.wire_bytes > 0
                    and pending.wire_bytes + prepared.wire_bytes > max_file_bytes
                ):
                    await flush_pending()
                if pending is None:
                    pending = _PendingDeltaFile.empty(is_seed=prepared.is_seed)
                pending.add(prepared, dtype_str)
                if pending.wire_bytes >= max_file_bytes:
                    await flush_pending()
            torch.distributed.barrier()

        if rank == 0:
            await flush_pending()

    async def _send_one_chunk_native(self, chunk: WeightChunk, version: int, file_index: int) -> None:
        dtype_str = self._init_info.model_dtype_str
        prepared = self._prepare_delta_chunk(chunk)
        payload = prepared.payload
        checksum = delta_checksum(None if prepared.is_seed else payload.positions, payload.values)
        positions_dtype = self._init_info.positions_transfer_dtype
        if not prepared.is_seed:
            assert payload.positions is not None
            positions_dtype = position_dtype_name(payload.positions.dtype)
        update_info = build_delta_manifest(
            names=prepared.names,
            dtype_names=[dtype_str] * len(prepared.names),
            shapes=prepared.shapes,
            counts=prepared.counts,
            is_seed=prepared.is_seed,
            checksum=checksum,
            version=version,
            file_index=file_index,
            positions_dtype=positions_dtype,
        )

        assert self._transport is not None
        if self._transport.sends_during_rpc:
            # NCCL: the rank-0 broadcast rendezvous with each worker's receive *inside* the
            # apply RPC, so launch the RPC first, then broadcast concurrently.
            update_task = asyncio.create_task(self._inference_client.update_weights_nccl(update_info))
            await asyncio.to_thread(self._transport.send, payload, version=version, file_index=file_index)
            await update_task
        else:
            # Disk: the file must be durable before the worker reads it during the apply RPC,
            # so write+fsync first, then issue the RPC.
            await asyncio.to_thread(self._transport.send, payload, version=version, file_index=file_index)
            await self._inference_client.update_weights_nccl(update_info)

    def teardown(self) -> None:
        if self._transport is not None:
            self._transport.teardown()
            self._transport = None
        self._differ.reset()


class DeltaTransferStrategy(WeightTransferStrategy):
    """Factory for delta (sparse bf16) weight transfer (new-inference path only)."""

    @staticmethod
    def create_init_info(ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None) -> DeltaInitInfo:
        # Delta sync is new-inference only: the receiver is vLLM's DeltaWeightTransferEngine,
        # so there is no legacy SkyRL-receiver code path to fall back to.
        if not _SKYRL_USE_NEW_INFERENCE:
            raise NotImplementedError("Delta weight sync requires the new inference path (_SKYRL_USE_NEW_INFERENCE=1).")
        # New inference path: world_size is fetched from the running servers.
        if inference_world_size is None:
            raise ValueError("inference_world_size must be provided when using new inference path")
        world_size = inference_world_size + 1  # +1 for trainer rank 0

        master_addr = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        # `weight_sync_backend == "delta"` selects this strategy; `delta_weight_sync_config`
        # picks the payload carrier (NCCL broadcast or shared-FS disk). The NCCL group params
        # below are always populated but only used by the NCCL transport.
        delta_cfg = ie_cfg.delta_weight_sync_config
        return DeltaInitInfo(
            master_addr=master_addr,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
            group_name="skyrl_delta",
            backend="nccl",
            model_dtype_str=ie_cfg.model_dtype,
            transport=delta_cfg.transport,
            sync_dir=delta_cfg.sync_dir,
            max_file_size_in_gb=delta_cfg.max_file_size_in_gb,
            max_files_to_keep=delta_cfg.max_files_to_keep,
            positions_transfer_dtype=delta_cfg.positions_transfer_dtype,
            trainer_diff_stage_area=delta_cfg.trainer_diff_stage_area,
            diff_num_workers=delta_cfg.diff_num_workers,
            override_existing_receiver=ie_cfg.override_existing_update_group == "enable",
        )

    @staticmethod
    def _build_sender_transport(init_info: DeltaInitInfo) -> DeltaTransport:
        """Build the rank-0 sender transport for the new inference path."""
        if init_info.transport == "nccl":
            # Join the same PyNccl group vLLM's engine uses (trainer = rank 0).
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )

            group = NCCLWeightTransferEngine.trainer_init(
                dict(
                    master_address=init_info.master_addr,
                    master_port=init_info.master_port,
                    world_size=init_info.world_size,
                )
            )
            device = torch.device("cuda", torch.cuda.current_device())
            return NcclDeltaTransport(group, device)
        if init_info.transport == "disk":
            if not init_info.sync_dir:
                raise ValueError("delta transport='disk' requires delta_weight_sync_config.sync_dir")
            return DiskDeltaTransport(init_info.sync_dir, max_files_to_keep=init_info.max_files_to_keep)
        raise ValueError(f"Unsupported delta transport {init_info.transport!r}; expected 'nccl' or 'disk'.")

    @staticmethod
    def create_sender(
        init_info: DeltaInitInfo,
        inference_client: InferenceEngineClient,
    ) -> DeltaWeightTransferSender:
        # Only rank 0 diffs/sends, so only it builds the payload transport.
        transport = None
        if torch.distributed.get_rank() == 0:
            transport = DeltaTransferStrategy._build_sender_transport(init_info)
        return DeltaWeightTransferSender(
            init_info=init_info,
            inference_client=inference_client,
            transport=transport,
        )

    @staticmethod
    def create_receiver(init_info: DeltaInitInfo) -> WeightTransferReceiver:
        # The delta receiver is vLLM's DeltaWeightTransferEngine, instantiated by vLLM's
        # weight-transfer factory on the inference worker (new inference path) -- not via this
        # SkyRL-side hook. Delta sync does not support the legacy SkyRL-receiver path.
        raise NotImplementedError(
            "Delta weight sync has no SkyRL-side receiver; it runs on the new inference path "
            "where vLLM's DeltaWeightTransferEngine is the receiver."
        )
