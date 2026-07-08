"""Core helpers for delta weight sync (sharded bf16 shadow).

Delta weight sync ships only the *changed* bf16 elements over the (potentially cross-DC)
wire instead of full tensors. The trainer keeps a bf16 snapshot of what it last sent and
diffs against it (:class:`SnapshotDiffer`); the inference side keeps a **per-rank bf16
shadow of its local shard** (:class:`ShardShadow`) -- *not* a full unsharded copy -- and
applies the sparse delta on top of it before re-quantizing.

Why a sharded shadow (and not a full one): quantized inference runs in fp8/int, and
re-quantization (``process_weights_after_loading``) is block-non-local -- changing one
element rescales a whole block -- so we must re-quant from the *full current shard* in
bf16, not just the changed elements. Keeping that shard (1/TP of the model per worker) is
exactly enough; a full bf16 copy per worker would waste TP-1 shards' worth of memory.

How the shard is maintained without re-deriving TP sharding ourselves (which differs per
architecture -- column/row/fused-QKV/MoE): we *reuse* vLLM's own load path. The receiver
reconstructs a transient **full** tensor that is ``NaN`` everywhere except the changed
positions, and hands it to ``reload_weights``. vLLM's per-parameter ``weight_loader``
narrows full->shard as usual; :class:`ShardShadow`'s hooks patch the reload so that
(a) ``Tensor.copy_`` is NaN-masked against the pinned-CPU shard shadow (the previous shard,
the "replay" baseline), so only the changed shard positions are overwritten, and (b) the
updated bf16 shard is **persisted back to the pinned-CPU shadow** just before
``process_weights_after`` re-quants it to fp8/int. The persistent shadow therefore lives on
host RAM, never GPU.

See ``vllm/ai_notes/05_delta_loadweights_cpu_shadow_review.md`` for the full design.

The diff / pack / build / mask primitives below are deliberately ``torch``-only (no vLLM,
no ``torch.distributed``) so they can be unit-tested on CPU. The only vLLM coupling lives
in :meth:`ShardShadow.install_persistent` (lazy ``vllm`` import), exercised on the worker.
"""

from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

# Positions index into the *flattened* full tensor. PyTorch indexing APIs are safest with
# int64, but the wire can use int32 for tensors that cannot overflow it.
POSITION_DTYPE = torch.int64
NARROW_POSITION_DTYPE = torch.int32
POSITION_INT32_MAX = torch.iinfo(torch.int32).max


def position_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.int64:
        return "int64"
    raise ValueError(f"Unsupported delta position dtype: {dtype}")


def position_dtype_from_name(name: str) -> torch.dtype:
    if name == "int32":
        return torch.int32
    if name == "int64":
        return torch.int64
    raise ValueError(f"Unsupported delta position dtype: {name!r}")


def validate_position_dtype_for_shapes(
    names: Sequence[str],
    shapes: Sequence[Sequence[int]],
    position_dtype: torch.dtype,
) -> None:
    """Validate that flat per-tensor positions fit in the configured transfer dtype."""
    if position_dtype == torch.int64:
        return
    if position_dtype != torch.int32:
        raise ValueError(f"Unsupported delta position dtype: {position_dtype}")

    for name, shape in zip(names, shapes):
        numel = 1
        for dim in shape:
            numel *= int(dim)
        if numel > POSITION_INT32_MAX + 1:
            raise ValueError(
                "delta_weight_sync_config.positions_transfer_dtype='int32' cannot encode "
                f"positions for parameter {name!r} with shape {list(shape)} ({numel} elements). "
                "Set delta_weight_sync_config.positions_transfer_dtype='int64' for this model."
            )


class SnapshotDiffer:
    """Trainer-side state: the bf16 snapshot of the last weights sent per parameter.

    For each parameter the first call is a *seed* (dense -- the whole tensor is "changed"),
    and subsequent calls return only the elements that differ from the snapshot. The
    snapshot is advanced to the current values on every call, so it always mirrors what the
    receiver's shard shadow reflects (assuming lossless delivery).
    """

    def __init__(self) -> None:
        # name -> flat bf16 snapshot on CPU
        self._snapshot: Dict[str, torch.Tensor] = {}

    def has(self, name: str) -> bool:
        return name in self._snapshot

    @staticmethod
    def _raise_if_nan(name: str, cur: torch.Tensor) -> None:
        if bool(torch.isnan(cur).any().item()):
            raise ValueError(
                f"Parameter '{name}' contains NaN values; refusing to sync (NaN is the "
                f"delta 'unchanged' sentinel and would not propagate). This usually means "
                f"training diverged."
            )

    @staticmethod
    def _to_flat_bf16(name: str, full: torch.Tensor) -> torch.Tensor:
        """Flatten to CPU bf16 and fail fast on NaN.

        NaN is the "unchanged" sentinel on the wire (the receiver skips NaN positions when
        applying a delta), so a genuinely-NaN trained weight would silently never propagate.
        That almost always means training diverged -- error out at the weight update rather
        than ship a corrupt/garbage update.
        """
        cur = full.detach().to(torch.bfloat16).reshape(-1).cpu()
        SnapshotDiffer._raise_if_nan(name, cur)
        return cur

    def seed(self, name: str, full: torch.Tensor) -> torch.Tensor:
        """Force a (re)seed: store the snapshot and return the dense flat bf16 values."""
        cur = self._to_flat_bf16(name, full)
        self._snapshot[name] = cur.clone()
        return cur

    def seed_many(
        self,
        names: Sequence[str],
        tensors: Sequence[torch.Tensor],
        *,
        num_workers: int = 1,
    ) -> List[torch.Tensor]:
        """Seed a chunk, optionally flattening tensors on multiple CPU worker threads."""
        if num_workers <= 0:
            num_workers = len(names)
        if num_workers <= 1 or len(names) <= 1:
            return [self.seed(n, t) for n, t in zip(names, tensors)]

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            values = list(pool.map(lambda nt: self._to_flat_bf16(*nt), zip(names, tensors)))

        for name, cur in zip(names, values):
            self._snapshot[name] = cur.clone()
        return values

    def diff(self, name: str, full: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor], torch.Tensor]:
        """Diff ``full`` against the snapshot for ``name`` and advance the snapshot.

        Returns ``(is_seed, positions, values)``:
          * seed:  ``(True,  None, flat_values)`` -- the whole (flattened) tensor.
          * delta: ``(False, positions, values)`` -- only changed flat indices/values.
        """
        cur = self._to_flat_bf16(name, full)
        if name not in self._snapshot:
            self._snapshot[name] = cur.clone()
            return True, None, cur
        prev = self._snapshot[name]
        if prev.numel() != cur.numel():
            # Shape changed (shouldn't happen mid-run); reseed.
            self._snapshot[name] = cur.clone()
            return True, None, cur
        mask = prev != cur
        positions = mask.nonzero(as_tuple=False).flatten().to(POSITION_DTYPE)
        values = cur[positions]
        self._snapshot[name] = cur.clone()
        return False, positions, values

    def diff_gpu(self, name: str, full: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor], torch.Tensor]:
        """Diff on CUDA, keeping the persistent snapshot on CPU.

        This path is useful when the extractor already yields CUDA tensors: it compares the
        current tensor against a transient GPU copy of the previous CPU snapshot, then copies
        only sparse ``positions``/``values`` back to CPU and patches the CPU snapshot in-place.
        It avoids the current CPU path's full current-tensor D2H copy for every delta, at the
        cost of a full previous-snapshot H2D copy for comparison.
        """
        if not full.is_cuda:
            return self.diff(name, full)

        cur = full.detach().to(torch.bfloat16).reshape(-1)
        self._raise_if_nan(name, cur)
        if name not in self._snapshot:
            cur_cpu = cur.cpu()
            self._snapshot[name] = cur_cpu.clone()
            return True, None, cur_cpu

        prev = self._snapshot[name]
        if prev.numel() != cur.numel():
            cur_cpu = cur.cpu()
            self._snapshot[name] = cur_cpu.clone()
            return True, None, cur_cpu

        prev_gpu = prev.to(device=cur.device, non_blocking=True)
        mask = prev_gpu != cur
        positions_gpu = mask.nonzero(as_tuple=False).flatten()
        values_gpu = cur[positions_gpu]
        positions = positions_gpu.to(POSITION_DTYPE).cpu()
        values = values_gpu.cpu()
        if positions.numel() > 0:
            prev[positions] = values
        return False, positions, values

    def diff_many(
        self,
        names: Sequence[str],
        tensors: Sequence[torch.Tensor],
        *,
        backend: str = "cpu",
        num_workers: int = 1,
    ) -> List[Tuple[bool, Optional[torch.Tensor], torch.Tensor]]:
        """Diff a chunk, optionally parallelizing per-parameter CPU work.

        ``backend="cpu"`` copies current weights to CPU before comparing. ``backend="gpu"``
        compares CUDA tensors against a transient GPU copy of the previous CPU snapshot and
        copies only sparse positions/values back. ``num_workers <= 0`` means one worker per
        tensor in the chunk.
        """
        if backend != "cpu":
            if backend != "gpu":
                raise ValueError(f"Unsupported delta diff backend {backend!r}; expected 'cpu' or 'gpu'.")
        if num_workers <= 0:
            num_workers = len(names)
        if num_workers <= 1 or len(names) <= 1:
            if backend == "gpu":
                return [self.diff_gpu(n, t) for n, t in zip(names, tensors)]
            return [self.diff(n, t) for n, t in zip(names, tensors)]

        if backend == "gpu":
            # Each WeightChunk entry has an independent parameter name/snapshot, so the GPU
            # path can update each parameter's CPU snapshot from its own worker. This keeps
            # sparse CPU snapshot patching parallel with the GPU compare work.
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                return list(pool.map(lambda nt: self.diff_gpu(*nt), zip(names, tensors)))

        prevs = [self._snapshot.get(n) for n in names]

        def compute_cpu(args: Tuple[str, torch.Tensor, Optional[torch.Tensor]]):
            n, t, prev = args
            cur = self._to_flat_bf16(n, t)
            if prev is None or prev.numel() != cur.numel():
                return True, None, cur, cur.clone()
            mask = prev != cur
            positions = mask.nonzero(as_tuple=False).flatten().to(POSITION_DTYPE)
            values = cur[positions]
            return False, positions, values, cur.clone()

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(compute_cpu, zip(names, tensors, prevs)))

        diffs: List[Tuple[bool, Optional[torch.Tensor], torch.Tensor]] = []
        for name, (is_seed, positions, values, snapshot) in zip(names, results):
            self._snapshot[name] = snapshot
            diffs.append((is_seed, positions, values))
        return diffs

    def reset(self, name: Optional[str] = None) -> None:
        """Forget the snapshot (forces a reseed). Drops one param or all."""
        if name is None:
            self._snapshot.clear()
        else:
            self._snapshot.pop(name, None)


# --- packing helpers (device-agnostic; used by the broadcast transport) ----------------


def pack_seed(values_per_param: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    """Concatenate dense flat values; return ``(packed_values, counts)``."""
    counts = [int(v.numel()) for v in values_per_param]
    if not values_per_param:
        return torch.empty(0, dtype=torch.bfloat16), counts
    packed = torch.cat([v.detach().to(torch.bfloat16).reshape(-1) for v in values_per_param])
    return packed, counts


def pack_delta(
    diffs_per_param: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    position_dtype: torch.dtype = POSITION_DTYPE,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Concatenate sparse ``(positions, values)`` pairs.

    Returns ``(packed_positions, packed_values, counts)`` where ``counts[i]`` is the number
    of changed elements for parameter ``i``.
    """
    if position_dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Unsupported delta position dtype: {position_dtype}")
    counts = [int(pos.numel()) for pos, _ in diffs_per_param]
    if not diffs_per_param:
        return (
            torch.empty(0, dtype=position_dtype),
            torch.empty(0, dtype=torch.bfloat16),
            counts,
        )
    packed_pos = torch.cat([pos.to(position_dtype).reshape(-1) for pos, _ in diffs_per_param])
    packed_val = torch.cat([val.to(torch.bfloat16).reshape(-1) for _, val in diffs_per_param])
    return packed_pos, packed_val, counts


def iter_unpack(counts: List[int]) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(index, offset, count)`` slices for unpacking a packed buffer."""
    offset = 0
    for i, count in enumerate(counts):
        yield i, offset, count
        offset += count


# --- integrity checksum (transport-agnostic) ------------------------------------------


def _crc32_update(crc: int, t: Optional[torch.Tensor]) -> int:
    """Fold ``t``'s raw bytes into a running CRC32 (no-op for ``None``/empty)."""
    if t is None or t.numel() == 0:
        return crc
    flat = t.detach().reshape(-1).contiguous().cpu()
    # ``view(uint8)`` reinterprets the bytes for any dtype (bf16 has no numpy dtype, so a
    # plain ``.numpy()`` would fail); the resulting uint8 array exposes the buffer protocol.
    return zlib.crc32(flat.view(torch.uint8).numpy(), crc)


def delta_checksum(positions: Optional[torch.Tensor], values: torch.Tensor) -> int:
    """CRC32 over the packed payload bytes (positions then values).

    Transport-agnostic integrity guard: the trainer computes it over the exact bytes it
    ships and embeds it in the (control-plane) manifest; the receiver recomputes it over the
    bytes it obtained -- via NCCL broadcast or a disk read -- and compares before applying.
    Catches silent corruption on the bulk-data channel independently of the reliable RPC.
    Seeds pass ``positions=None`` (dense, no positions on the wire).
    """
    crc = _crc32_update(0, positions)
    crc = _crc32_update(crc, values)
    return crc & 0xFFFFFFFF


def verify_delta_checksum(expected: int, positions: Optional[torch.Tensor], values: torch.Tensor) -> None:
    """Raise if the received payload's CRC32 does not match ``expected``."""
    actual = delta_checksum(positions, values)
    if actual != (int(expected) & 0xFFFFFFFF):
        raise ValueError(
            f"Delta payload checksum mismatch: expected {int(expected) & 0xFFFFFFFF:#010x}, "
            f"got {actual:#010x}. The bulk-data channel (NCCL/disk) likely corrupted the payload."
        )


# --- unified delta-parameter manifest --------------------------------------------------


@dataclass
class DeltaParam:
    """One parameter's entry in a delta chunk manifest (transport-agnostic).

    A chunk concatenates the sparse payloads of several params into one packed
    ``(positions, values)`` buffer; ``count`` is this param's slice length within that
    buffer (equal to ``prod(shape)`` for a seed, the number of changed elements otherwise).
    This is the single per-param description shared by sender and receiver, regardless of
    whether the payload travels over NCCL or disk.
    """

    name: str
    dtype_name: str
    shape: List[int]
    count: int


def build_delta_manifest(
    names: List[str],
    dtype_names: List[str],
    shapes: List[List[int]],
    counts: List[int],
    *,
    is_seed: bool,
    checksum: int,
    version: int = 0,
    file_index: int = 0,
    positions_dtype: str = "int64",
) -> Dict[str, Any]:
    """Assemble the flat control-plane manifest dict for one delta file.

    The dict crosses the (reliable) RPC unchanged for every transport; the bulk
    ``(positions, values)`` payload is carried separately by the transport. ``version`` /
    ``file_index`` route a disk read to the right file and are ignored by NCCL.
    """
    return {
        "names": list(names),
        "dtype_names": list(dtype_names),
        "shapes": [list(s) for s in shapes],
        "counts": list(counts),
        "is_seed": is_seed,
        "checksum": int(checksum),
        "version": int(version),
        "file_index": int(file_index),
        "positions_dtype": positions_dtype,
    }


def iter_delta_params(
    names: List[str], dtype_names: List[str], shapes: List[List[int]], counts: List[int]
) -> Iterator[DeltaParam]:
    """Yield a :class:`DeltaParam` per parameter from the aligned manifest lists."""
    for name, dtype_name, shape, count in zip(names, dtype_names, shapes, counts):
        yield DeltaParam(name=name, dtype_name=dtype_name, shape=list(shape), count=int(count))


# --- full-tensor builders (the transient "NaN-narrow" payload) -------------------------


def build_full_seed(
    shape: List[int],
    values: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build the dense full tensor for a seed (every element present)."""
    return values.detach().to(dtype).reshape(*shape).to(device) if device else values.detach().to(dtype).reshape(*shape)


def build_full_delta(
    shape: List[int],
    positions: torch.Tensor,
    values: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build a full tensor that is ``NaN`` everywhere except the changed ``positions``.

    The ``NaN`` sentinel is what :class:`ShardShadow`'s patched ``copy_`` keys on: when vLLM
    narrows this full tensor to the local shard and copies it in, the NaN-masked ``copy_``
    overwrites only the changed shard positions, leaving the rest at the previous shard
    value (held by the shadow).
    """
    numel = 1
    for s in shape:
        numel *= int(s)
    flat = torch.full((numel,), float("nan"), dtype=dtype, device=device)
    if positions.numel() > 0:
        flat.index_copy_(
            0,
            positions.to(POSITION_DTYPE).to(flat.device),
            values.detach().to(dtype).reshape(-1).to(flat.device),
        )
    return flat.view(*shape)


def _storage_ptr(t: torch.Tensor) -> int:
    """Base storage address of ``t`` (shared by a tensor and its slices/views)."""
    try:
        return t.untyped_storage().data_ptr()
    except Exception:  # pragma: no cover - very old torch
        return t.storage().data_ptr()


# --- pre-quantization bf16 master capture (load-time shadow priming) -------------------
#
# A quantized inference model has no bf16 master available after load: the weights are cast
# to fp8/kernel format and bf16<-fp8 is lossy. The bf16 master exists for exactly one
# instant per layer -- right after its weight loaders run, right *before* its
# ``quant_method.process_weights_after_loading`` quantizes it. We snapshot it there (driven
# by import-time wraps in the worker wrap, see ``new_inference_worker_wrap``).
#
# vLLM has *two* such bf16->kernel boundaries depending on the quant method:
#   - meta-device "online" quant (``quantization=fp8`` etc., ``uses_meta_device=True``):
#     the cast happens per layer inside ``reload.layerwise._layerwise_process``, *before*
#     the model-level ``process_weights_after_loading`` runs;
#   - non-meta "quantize in postprocess": the cast happens in the model-level
#     ``base_loader.process_weights_after_loading``.
# Both feed this single mechanism, so it generalizes across quant strategies without any
# per-quant-method knowledge.
#
# The stash is keyed by *module identity* ``(id(module), local_param_name)`` rather than the
# full param name, because the capture seams see a layer (or a still-being-built model) but
# not the full module path. :meth:`ShardShadow.initialize` resolves these to full names via
# the live module tree (module identity is stable across materialization) and adopts them as
# the bf16 master. One model per vLLM worker process, so a per-process global suffices.
#
# This is also what lets the disk transport skip the initial full-weight "seed" even for a
# quantized engine: the receiver reconstructs the true bf16 base locally instead of having
# the trainer ship it.
_PREQUANT_BF16: Dict[Tuple[int, str], torch.Tensor] = {}


def capture_layer_prequant_bf16(layer: torch.nn.Module) -> None:
    """Snapshot ``layer``'s bf16 weight shards into the global stash, keyed by module identity.

    Call *before* the layer's ``process_weights_after_loading`` quantizes its (already loaded)
    weights, so the params are still the bf16 master. The inference model is already
    TP-sharded, so each captured tensor is this worker's local shard.

    Only ``torch.bfloat16`` params are captured -- an already-fp8 param (a pre-quantized
    checkpoint with no on-the-fly conversion) has no bf16 master to record. Idempotent per
    ``(layer, param)``: a key already present is left untouched (so re-entrant or duplicate
    calls never clobber the first, true pre-quant snapshot).
    """
    orig_copy = torch.Tensor.copy_
    for local_name, param in layer._parameters.items():
        if param is None or param.is_meta or param.dtype != torch.bfloat16:
            continue
        key = (id(layer), local_name)
        if key in _PREQUANT_BF16:
            continue
        _PREQUANT_BF16[key] = ShardShadow._snapshot(param.data, orig_copy)


def capture_model_prequant_bf16(model: torch.nn.Module) -> None:
    """Capture every module's bf16 weights (the non-meta ``process_weights_after_loading``
    seam, where all params are still bf16 model-wide). Thin per-module fan-out of
    :func:`capture_layer_prequant_bf16`."""
    for module in model.modules():
        capture_layer_prequant_bf16(module)


def take_prequant_bf16() -> Dict[Tuple[int, str], torch.Tensor]:
    """Hand the captured masters off to the shadow and clear the stash.

    Returns ``{}`` when nothing was captured (a bf16 model loaded without the wraps, or a
    fully pre-quantized checkpoint). Keyed by ``(id(module), local_param_name)``; the caller
    (:meth:`ShardShadow.initialize`) resolves these to full param names via the live model.
    """
    global _PREQUANT_BF16
    captured = _PREQUANT_BF16
    _PREQUANT_BF16 = {}
    return captured


def reset_prequant_capture() -> None:
    """Test-only: forget any captured masters."""
    global _PREQUANT_BF16
    _PREQUANT_BF16 = {}


class ShardShadow:
    """Inference-side state: the bf16 master of *this worker's shard* per parameter.

    The shadow is a ``{full_param_name -> pinned-CPU bf16 master}`` map -- one entry per whole
    (post-fusion) param holding this worker's bf16 shard (1/TP of the model). A fused param
    such as ``qkv_proj.weight`` is a single entry; its q/k/v ``weight_loader`` slices are
    addressed into that entry with ``as_strided`` at ``copy_`` time. Keeping the whole param
    (rather than per-slice keys discovered lazily during the seed) is what lets the master be
    primed **explicitly** up front via :meth:`initialize`, so every delta -- including the
    first -- always has a finite merge base. The master lives in **pinned host memory**, so it
    never occupies GPU memory; only a transient bf16 shard lives on GPU during a reload.

    Two seams:

    * **Persistent (per training run)** -- :meth:`install_persistent`: a thin
      ``layerwise.materialize_layer`` patch that, for each just-materialized param, records
      ``base_storage_ptr -> full_name`` (registration only -- no staging). Inert unless an
      update is :attr:`active`.
    * **Per weight update** -- :meth:`begin_update` / :meth:`end_update`: the global
      ``torch.Tensor.copy_`` patch. For every tracked ``copy_`` (a ``weight_loader`` writing a
      bf16 slice, *before* ``process_weights_after_loading`` quantizes it) it: (a) NaN-merges
      the narrowed delta against the corresponding slice of the bf16 master (via ``as_strided``)
      so only changed positions overwrite, and (b) D2H-persists the merged bf16 slice back into
      the master. ``copy_`` is a hot global monkeypatch, so it is only live for the brief
      reload window.

    Because persistence happens at ``copy_`` (which is exactly the bf16 input to re-quant),
    there is **no** ``process_weights_after_loading`` wrap -- both quant and no-quant params
    are captured uniformly at ``copy_``.

    Keying note: ``copy_`` sees only ``(dest, src)``; the destination's param *name* is resolved
    via the materialize-time ``storage_ptr -> name`` registration (storage only exists after
    ``materialize_layer``), and the exact slice within the master is recovered from the
    destination view's ``(size, stride, storage_offset)``. The whole sync runs inside a single
    :meth:`begin_update` / :meth:`end_update` window (one ``initialize_layerwise_reload`` ->
    ``finalize_layerwise_reload`` pass), so every tracked write is materialized and copied in the
    same window -- there are no cross-window writes to chase.

    The bf16 master is primed explicitly by :meth:`initialize` (called from
    :meth:`install_persistent`), so a delta's merge base never depends on the seed ``copy_``
    having been observed. The delta sender still yields every param each sync, but an unwritten
    slice simply retains its master value.

    Use :meth:`install` / :meth:`uninstall` for the split start/finish lifecycle (the only
    inference path), and :meth:`shutdown` (or :meth:`uninstall_persistent`) at receiver teardown.
    """

    def __init__(self) -> None:
        # full_param_name -> pinned-CPU bf16 master of *this worker's whole shard* of the
        # param (survives across updates). Keyed by the whole (post-fusion) param so it can be
        # primed explicitly via :meth:`initialize` without knowing each loader's slice layout;
        # individual fused/sharded slices are addressed into it with ``as_strided`` at copy time.
        self._shard: Dict[str, torch.Tensor] = {}
        # --- persistent (per-run) install state ---
        self._persistent: bool = False
        self._layerwise: Optional[Any] = None
        self._orig_copy: Optional[Callable] = None
        self._orig_materialize: Optional[Callable] = None
        self._name_map: Dict[Tuple[int, str], str] = {}
        # --- per-update state (reset by `begin_update`) ---
        self._active: bool = False
        # base storage_ptr -> full_name for params materialized this update. Storage is
        # re-allocated each reload, so this is rebuilt every begin_update().
        self._name_by_storage: Dict[int, str] = {}

    def has(self, name: str) -> bool:
        return name in self._shard

    def names(self) -> List[str]:
        return list(self._shard.keys())

    def clear(self) -> None:
        self._shard.clear()

    def shard(self, name: str) -> torch.Tensor:
        return self._shard[name]

    @property
    def initialized(self) -> bool:
        """True once the shadow has been primed with a bf16 master (see :meth:`initialize`)."""
        return bool(self._shard)

    @property
    def installed(self) -> bool:
        """True once the persistent (per-run) hooks are installed."""
        return self._persistent

    @property
    def active(self) -> bool:
        """True between :meth:`begin_update` and :meth:`end_update`."""
        return self._active

    @staticmethod
    def _build_name_map(model: torch.nn.Module) -> Dict[Tuple[int, str], str]:
        """Map ``(id(module), local_param_name) -> full_param_name`` for all params/buffers.

        Built from the *live* module tree (stable across materialization, which replaces
        tensors but keeps module identity).
        """
        name_map: Dict[Tuple[int, str], str] = {}
        for module_name, module in model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for local_name, _ in module.named_parameters(recurse=False):
                name_map[(id(module), local_name)] = prefix + local_name
            for local_name, _ in module.named_buffers(recurse=False):
                name_map[(id(module), local_name)] = prefix + local_name
        return name_map

    @staticmethod
    def _snapshot(src: torch.Tensor, orig_copy: Callable) -> torch.Tensor:
        """Make a contiguous pinned-CPU bf16 copy of ``src`` (the whole-param shard)."""
        src = src.detach()
        pin = torch.cuda.is_available()
        cpu = torch.empty(tuple(src.shape), dtype=torch.bfloat16, device="cpu", pin_memory=pin)
        orig_copy(cpu, src.to(torch.bfloat16))
        return cpu

    def initialize(self, model: torch.nn.Module, *, force: bool = False) -> None:
        """Explicitly prime the shadow with this worker's bf16 shard (the master).

        For each floating param the master is, in priority order:

        1. the **pre-quantization bf16** captured at load time (by
           :func:`capture_layer_prequant_bf16`, keyed by module identity and resolved to the
           full name here) -- the only correct base for a quantized param, whose live value is
           now fp8;
        2. otherwise, a live snapshot of the param if it is *still bf16* -- correct for a bf16
           model and for the unquantized params (norms/embeddings) of a quantized model, which
           never go through a quant cast.

        A param that is neither captured nor live-bf16 (a genuinely pre-quantized fp8
        checkpoint with no transient bf16 stage) gets **no** master: delta sync cannot
        reconstruct a base for it locally, so that setup must ship an explicit seed. This is
        the inherent limit, not a gap to backfill.

        Priming guarantees every *supported* param has a finite bf16 merge base, so a delta's
        ``NaN`` positions always fall back to a real value. After priming, deltas accumulate
        **in the shadow** (never re-read from the live, possibly-quantized param).

        Idempotent: a no-op once primed unless ``force=True``. Must run while no update is
        active (so the global ``copy_`` patch is not installed).
        """
        if self._shard and not force:
            return
        if self._active:
            raise RuntimeError("initialize() must run outside an active update")
        # Captured masters are keyed by module identity; resolve to full names via the live
        # module tree (identity is stable across materialization).
        captured = take_prequant_bf16()
        orig_copy = self._orig_copy or torch.Tensor.copy_
        self._shard.clear()
        for module_name, module in model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for local_name, param in module._parameters.items():
                if param is None or not param.is_floating_point():
                    continue
                cap = captured.get((id(module), local_name))
                if cap is not None:
                    # True pre-quant bf16 master snapshotted before the fp8 cast.
                    self._shard[prefix + local_name] = cap
                elif not param.is_meta and param.dtype == torch.bfloat16:
                    # Still-bf16 param (bf16 model, or an unquantized param of a quantized
                    # model): the live shard is the master. Skip meta params (the new path
                    # meta-izes during reload) and already-quantized params (no bf16 master).
                    self._shard[prefix + local_name] = self._snapshot(param.data, orig_copy)

    def _shadow_slice(self, name: str, ref: torch.Tensor) -> Optional[torch.Tensor]:
        """View into the whole-param shadow matching ``ref``'s slice of the live param.

        ``ref`` is a (possibly narrowed) view of the freshly materialized param; the shadow is
        a contiguous bf16 tensor with the same full shape, so the same
        ``(size, stride, storage_offset)`` addresses the same logical elements. Returns ``None``
        if the param was not primed or the slice would fall outside the shadow storage.
        """
        shadow_full = self._shard.get(name)
        if shadow_full is None:
            return None
        try:
            return torch.as_strided(shadow_full, ref.size(), ref.stride(), ref.storage_offset())
        except RuntimeError:
            return None

    def _persist_slice(self, name: str, gpu_tensor: torch.Tensor) -> None:
        """D2H-persist ``gpu_tensor`` (the merged bf16 slice) back into the whole-param shadow."""
        assert self._orig_copy is not None
        src = gpu_tensor.detach()
        dst = self._shadow_slice(name, src)
        if dst is None:
            return
        self._orig_copy(dst, src.to(torch.bfloat16), non_blocking=True)

    # --- persistent (per-run) hooks ----------------------------------------------------

    def install_persistent(self, model: torch.nn.Module) -> None:
        """Install the per-run hook: a thin ``materialize_layer`` patch (registration only).

        Idempotent (no-op if already installed). The patch only records
        ``base_storage_ptr -> full_name`` for params materialized during an active update, so
        it is inert between updates (and during the initial model load).
        """
        if self._persistent:
            return
        # Lazy import: keeps this module importable (and unit-testable) without vLLM.
        from vllm.model_executor.model_loader.reload import layerwise as _layerwise

        self._layerwise = _layerwise
        self._name_map = self._build_name_map(model)
        # The true, unpatched ``copy_`` (captured before any per-update patch). Used by the
        # ``copy_`` patch, ``initialize``, and ``_persist_slice`` to read/write the master
        # without re-entering the NaN mask.
        self._orig_copy = torch.Tensor.copy_
        self._orig_materialize = _layerwise.materialize_layer

        sh = self
        orig_materialize = self._orig_materialize

        def patched_materialize(layer, info):
            orig_materialize(layer, info)
            if sh._active:
                sh._on_materialize(layer)

        _layerwise.materialize_layer = patched_materialize  # type: ignore[assignment]
        # Explicitly prime the bf16 master from the model's current shard, so the very first
        # delta has a valid merge base (the live weights are still the high-precision master at
        # this point; later deltas accumulate in the shadow). No-op if already primed.
        self.initialize(model)
        self._persistent = True

    def uninstall_persistent(self) -> None:
        """Restore the per-run hook. Ends an in-flight update first."""
        if not self._persistent:
            return
        if self._active:
            self.end_update()
        assert self._layerwise is not None and self._orig_materialize is not None
        self._layerwise.materialize_layer = self._orig_materialize  # type: ignore[assignment]
        self._layerwise = None
        self._orig_copy = None
        self._orig_materialize = None
        self._name_map = {}
        self._persistent = False

    # --- per-update hooks ---------------------------------------------------------------

    def begin_update(self) -> None:
        """Start a weight update: reset per-update state and install the ``copy_`` patch.

        The global ``torch.Tensor.copy_`` monkeypatch lives only for the update window so it
        never taxes inference between syncs. Requires the persistent hooks to be installed.
        """
        if not self._persistent:
            raise RuntimeError("install_persistent(model) must be called before begin_update()")
        if self._active:
            raise RuntimeError("a weight update is already active; call end_update() first")
        self._name_by_storage = {}

        sh = self
        orig_copy = self._orig_copy
        assert orig_copy is not None

        def patched_copy(self_t, src, *args, **kwargs):
            # The single per-update seam. ``copy_`` is global while an update is active, but we
            # only act on a *tracked* destination -- one whose storage the materialize hook
            # registered this window (a ``weight_loader`` writing a bf16 slice into its just-
            # materialized param, before ``process_weights_after_loading`` re-quantizes it).
            # Everything else falls straight through to the real ``copy_``: untracked scratch
            # writes, fp8 re-quant outputs (which target *new* storage), and the meta-param
            # ``CopyCounter`` pass (``initialize_layerwise_reload`` runs every loader once on the
            # still-meta param to count elements -- skipped via the device guard).
            #
            # Because we key on tracked names -- always bf16 loader writes -- there is no need to
            # guard the source/destination dtype: a tracked ``src`` is the bf16 delta, so
            # ``torch.isnan`` is always valid. This is also why no live-model name fallback is
            # needed: the whole sync is one ``begin_update`` window, so every write is
            # materialized (and registered) and copied in the same window.
            if isinstance(src, torch.Tensor) and self_t.device.type != "meta":
                name = sh._name_by_storage.get(_storage_ptr(self_t))
                if name is not None:
                    # (a) NaN-merge against the previous shard slice -- a view into the primed
                    # whole-param bf16 master. The freshly materialized GPU param is
                    # uninitialised, so the merge base must come from the shadow (not self_t).
                    # A seed carries no NaN and overwrites densely.
                    if src.numel() > 0 and torch.isnan(src).any():
                        base_view = sh._shadow_slice(name, self_t)

                        if base_view is not None:
                            base = base_view.to(self_t.device, non_blocking=True)
                        else:
                            # No master: only happens for a genuinely pre-quantized param
                            # (fp8 from load, never bf16), which delta sync cannot base
                            # locally. Zero-fill so the NaN positions stay finite -- under fp8
                            # a single NaN poisons the whole per-tensor scale -- and warn that
                            # this param needs an explicit seed instead.
                            base = torch.zeros_like(self_t, dtype=src.dtype, device=self_t.device)
                            try:
                                from loguru import logger as _lg

                                _lg.warning(
                                    "ShardShadow: no bf16 master for {!r}; this param has no "
                                    "local base (pre-quantized?) and needs a shipped seed",
                                    name,
                                )
                            except Exception:
                                pass
                        if src.device != self_t.device:
                            src = src.to(self_t.device)
                        # base is finite (real master or zero-fill), so no residual NaN remains.
                        src = torch.where(torch.isnan(src), base, src)
                    result = orig_copy(self_t, src, *args, **kwargs)
                    # (b) persist the merged bf16 slice (the input to re-quant) into the master.
                    sh._persist_slice(name, self_t)
                    return result
            return orig_copy(self_t, src, *args, **kwargs)

        torch.Tensor.copy_ = patched_copy  # type: ignore[method-assign,assignment]
        self._active = True

    def end_update(self) -> None:
        """Finish a weight update: remove the ``copy_`` patch.

        Nothing is persisted here -- merge + persist already happened inline at every tracked
        ``copy_``. The persistent materialize hook is left installed for the next update.
        """
        if not self._active:
            return
        assert self._orig_copy is not None
        torch.Tensor.copy_ = self._orig_copy  # type: ignore[method-assign,assignment]
        self._name_by_storage = {}
        self._active = False

    def _on_materialize(self, layer: torch.nn.Module) -> None:
        """materialize hook: register ``base_storage_ptr -> full_name`` for this layer's params.

        Registration only (no staging / no persist). The ``copy_`` patch resolves a tracked
        write to its full param name via this map, then addresses the exact slice into the
        whole-param master with the destination view's ``(size, stride, storage_offset)``.
        Views of a param share the param's base storage pointer, so a single registration per
        param covers all its fused slices.
        """
        for local_name, param in list(layer._parameters.items()):
            if param is None or not param.is_floating_point():
                continue
            full = self._name_map.get((id(layer), local_name))
            if full is None:
                continue
            self._name_by_storage[_storage_ptr(param.data)] = full

    # --- lifecycle facades --------------------------------------------------------------

    def install(self, model: torch.nn.Module) -> None:
        """Begin an update for the split start/finish lifecycle (new-inference path).

        Installs the persistent hooks (if needed) and starts the per-update window. Pair with
        :meth:`uninstall` at ``finish_weight_update``.
        """
        self.install_persistent(model)
        self.begin_update()

    def uninstall(self) -> None:
        """End the current update and remove all hooks (split start/finish lifecycle)."""
        self.end_update()
        self.uninstall_persistent()

    def shutdown(self) -> None:
        """Tear down all hooks (end any active update, uninstall persistent). Keeps shards."""
        self.uninstall_persistent()
