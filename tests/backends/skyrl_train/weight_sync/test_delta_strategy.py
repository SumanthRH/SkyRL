"""CPU unit tests for delta weight sync (sparse bf16 deltas + sharded shadow).

These cover the transport-independent core the new-inference ``DeltaWeightTransferEngine``
relies on:

* sender diff/pack -- ``SnapshotDiffer`` + ``pack_seed`` / ``pack_delta``;
* receiver reconstruction -- ``iter_unpack`` + ``build_full_seed`` / ``build_full_delta``
  (exactly what ``DeltaWeightTransferEngine.receive_weights`` feeds to ``load_weights``);
* NaN-masked shard replay -- the ``torch.where`` semantics ``ShardShadow``'s patched ``copy_``
  applies so only changed positions overwrite the previous shard;
* load-time bf16 capture -- ``capture_layer_prequant_bf16`` / ``capture_model_prequant_bf16``
  + ``ShardShadow.initialize`` adopting it (resolving module-identity keys to full names), the
  only correct master source for a quantized engine (the live params are already fp8).

They run on CPU with no distributed env and without importing vLLM. A separate, vLLM-guarded
test exercises the engine's update-info validation.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.delta_utils import (
    NARROW_POSITION_DTYPE,
    POSITION_DTYPE,
    POSITION_INT32_MAX,
    ShardShadow,
    SnapshotDiffer,
    _storage_ptr,
    build_full_delta,
    build_full_seed,
    capture_layer_prequant_bf16,
    capture_model_prequant_bf16,
    iter_unpack,
    pack_delta,
    pack_seed,
    reset_prequant_capture,
    take_prequant_bf16,
    validate_position_dtype_for_shapes,
)


@pytest.fixture(autouse=True)
def _reset_prequant_capture():
    """The pre-quant bf16 capture stash is a process-global; isolate it per test."""
    reset_prequant_capture()
    yield
    reset_prequant_capture()


NAMES = ["a.weight", "b.weight"]
SHAPES = [[4, 3], [5]]
CPU = torch.device("cpu")


def _rand_weights():
    return {n: torch.randn(*s, dtype=torch.bfloat16) for n, s in zip(NAMES, SHAPES)}


def _diff_and_pack(differ, curr):
    """Sender side: seed on first sight, else sparse diff. Returns the wire payload."""
    is_seed = not all(differ.has(n) for n in NAMES)
    if is_seed:
        values = [differ.seed(n, curr[n]) for n in NAMES]
        packed_values, counts = pack_seed(values)
        packed_positions = None
    else:
        diffs = []
        for n in NAMES:
            _, pos, val = differ.diff(n, curr[n])
            diffs.append((pos, val))
        packed_positions, packed_values, counts = pack_delta(diffs)
    return is_seed, counts, packed_values, packed_positions


def _reconstruct(counts, packed_values, packed_positions, is_seed):
    """Receiver side: mirror ``DeltaWeightTransferEngine.receive_weights`` (no broadcast)."""
    out = {}
    if is_seed:
        for i, offset, count in iter_unpack(counts):
            values = packed_values[offset : offset + count]
            out[NAMES[i]] = build_full_seed(SHAPES[i], values, device=CPU)
        return out
    for i, offset, count in iter_unpack(counts):
        positions = packed_positions[offset : offset + count]
        values = packed_values[offset : offset + count]
        out[NAMES[i]] = build_full_delta(SHAPES[i], positions, values, device=CPU)
    return out


def _replay(prev, full_payload):
    """Shadow side: NaN-masked merge of the (full) payload onto the previous shard.

    With TP=1 the local shard *is* the full tensor, so this models the engine end-to-end:
    seeds overwrite densely; deltas keep ``prev`` at NaN positions. Mirrors the ``torch.where``
    the shadow's patched ``copy_`` applies.
    """
    payload = full_payload.to(prev.dtype)
    return torch.where(torch.isnan(payload), prev, payload)


def test_seed_reconstruction_is_dense():
    torch.manual_seed(0)
    differ = SnapshotDiffer()
    w = _rand_weights()

    is_seed, counts, packed_values, packed_positions = _diff_and_pack(differ, w)
    assert is_seed
    assert counts == [12, 5]  # dense seed = full element counts

    payloads = _reconstruct(counts, packed_values, packed_positions, is_seed)
    for n in NAMES:
        assert not torch.isnan(payloads[n]).any()
        assert torch.equal(payloads[n], w[n])
        # Replaying a seed onto an (arbitrary) previous shard overwrites it densely.
        assert torch.equal(_replay(torch.zeros_like(w[n]), payloads[n]), w[n])


def test_delta_reconstruction_is_nan_narrow_and_replays_in_place():
    torch.manual_seed(1)
    differ = SnapshotDiffer()
    w = _rand_weights()
    _diff_and_pack(differ, w)  # seed

    w2 = {n: w[n].clone() for n in NAMES}
    w2["a.weight"][0, 0] = 9.0
    w2["a.weight"][2, 1] = -3.0

    is_seed, counts, packed_values, packed_positions = _diff_and_pack(differ, w2)
    assert not is_seed
    assert counts == [2, 0]  # only 2 changed in a.weight, 0 in b.weight

    payloads = _reconstruct(counts, packed_values, packed_positions, is_seed)

    # a.weight: NaN everywhere except the two changed positions (which hold the new values).
    a = payloads["a.weight"]
    changed = torch.zeros_like(a, dtype=torch.bool)
    changed[0, 0] = True
    changed[2, 1] = True
    assert torch.isnan(a[~changed]).all()
    assert not torch.isnan(a[changed]).any()
    assert a[0, 0].item() == 9.0
    assert a[2, 1].item() == -3.0

    # b.weight: untouched -> all NaN.
    assert torch.isnan(payloads["b.weight"]).all()

    # Replaying the deltas onto the previous shard reproduces the new full weights.
    assert torch.equal(_replay(w["a.weight"], payloads["a.weight"]), w2["a.weight"])
    assert torch.equal(_replay(w["b.weight"], payloads["b.weight"]), w["b.weight"])


def test_pack_delta_can_use_int32_positions():
    diffs = [
        (
            torch.tensor([0, 2], dtype=POSITION_DTYPE),
            torch.tensor([1.0, -1.0], dtype=torch.bfloat16),
        )
    ]

    packed_positions, packed_values, counts = pack_delta(diffs, position_dtype=NARROW_POSITION_DTYPE)

    assert counts == [2]
    assert packed_positions.dtype == torch.int32
    payload = build_full_delta([3], packed_positions, packed_values, device=CPU)
    assert payload[0].item() == 1.0
    assert torch.isnan(payload[1])
    assert payload[2].item() == -1.0


def test_int32_position_dtype_rejects_only_overflowing_params():
    validate_position_dtype_for_shapes(["ok"], [[POSITION_INT32_MAX + 1]], torch.int32)
    validate_position_dtype_for_shapes(["huge"], [[POSITION_INT32_MAX + 2]], torch.int64)
    with pytest.raises(ValueError, match="positions_transfer_dtype='int32'.*huge.*int64"):
        validate_position_dtype_for_shapes(["huge"], [[POSITION_INT32_MAX + 2]], torch.int32)


def test_no_change_delta_is_all_nan_and_replay_is_noop():
    torch.manual_seed(2)
    differ = SnapshotDiffer()
    w = _rand_weights()
    _diff_and_pack(differ, w)  # seed

    is_seed, counts, packed_values, packed_positions = _diff_and_pack(differ, w)  # identical
    assert not is_seed
    assert counts == [0, 0]

    payloads = _reconstruct(counts, packed_values, packed_positions, is_seed)
    for n in NAMES:
        assert torch.isnan(payloads[n]).all()
        # NaN-masked replay over the previous shard leaves it unchanged.
        assert torch.equal(_replay(w[n], payloads[n]), w[n])


def test_multiple_deltas_accumulate_on_shadow():
    torch.manual_seed(3)
    differ = SnapshotDiffer()
    w = _rand_weights()
    is_seed, counts, packed_values, packed_positions = _diff_and_pack(differ, w)
    # Maintain a running shard shadow (TP=1: full == shard), seeded densely.
    shadow = _reconstruct(counts, packed_values, packed_positions, is_seed)

    cur = {n: w[n].clone() for n in NAMES}
    for step in range(3):
        cur = {n: cur[n].clone() for n in NAMES}
        cur["b.weight"][step] = float(step + 1)
        is_seed, counts, packed_values, packed_positions = _diff_and_pack(differ, cur)
        assert not is_seed
        assert counts == [0, 1]
        payloads = _reconstruct(counts, packed_values, packed_positions, is_seed)
        shadow = {n: _replay(shadow[n], payloads[n]) for n in NAMES}
        for n in NAMES:
            assert torch.equal(shadow[n], cur[n])


def test_snapshot_differ_diff_many_cpu_parallel_matches_sequential():
    torch.manual_seed(4)
    names = [f"p{i}" for i in range(4)]
    base = [torch.randn(128, dtype=torch.bfloat16) for _ in names]
    cur = [t.clone() for t in base]
    cur[0][0:3] += 1
    cur[2][10] -= 2

    seq = SnapshotDiffer()
    par = SnapshotDiffer()
    seq.seed_many(names, base)
    par.seed_many(names, base, num_workers=2)

    seq_diffs = seq.diff_many(names, cur, backend="cpu", num_workers=1)
    par_diffs = par.diff_many(names, cur, backend="cpu", num_workers=0)

    for (seq_is_seed, seq_pos, seq_val), (par_is_seed, par_pos, par_val) in zip(seq_diffs, par_diffs):
        assert seq_is_seed is par_is_seed is False
        assert seq_pos is not None
        assert par_pos is not None
        assert torch.equal(seq_pos, par_pos)
        assert torch.equal(seq_val, par_val)
    for name in names:
        assert torch.equal(seq._snapshot[name], par._snapshot[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU delta diff")
def test_snapshot_differ_gpu_diff_matches_cpu():
    torch.manual_seed(5)
    names = ["a", "b"]
    base_gpu = [torch.randn(256, dtype=torch.bfloat16, device="cuda") for _ in names]
    cur_gpu = [t.clone() for t in base_gpu]
    cur_gpu[0][1:4] += 1
    cur_gpu[1][20] -= 2

    cpu = SnapshotDiffer()
    gpu = SnapshotDiffer()
    cpu.seed_many(names, base_gpu)
    gpu.seed_many(names, base_gpu)

    cpu_diffs = cpu.diff_many(names, cur_gpu, backend="cpu")
    gpu_diffs = gpu.diff_many(names, cur_gpu, backend="gpu", num_workers=0)

    for (cpu_is_seed, cpu_pos, cpu_val), (gpu_is_seed, gpu_pos, gpu_val) in zip(cpu_diffs, gpu_diffs):
        assert cpu_is_seed is gpu_is_seed is False
        assert cpu_pos is not None
        assert gpu_pos is not None
        assert torch.equal(cpu_pos, gpu_pos)
        assert torch.equal(cpu_val, gpu_val)
    for name in names:
        assert torch.equal(cpu._snapshot[name], gpu._snapshot[name])


def _active_shadow():
    """A ``ShardShadow`` with the per-update ``copy_`` patch live, sans vLLM.

    ``install_persistent`` imports vLLM only to patch ``materialize_layer`` (which records
    ``storage_ptr -> name``). Here we skip that and drive registration by hand so the v2
    ``begin_update`` ``copy_`` seam (merge + persist) can be tested on CPU. The caller MUST
    ``end_update()`` (restores the global ``copy_``); use the context manager below.
    """
    sh = ShardShadow()
    sh._persistent = True
    sh._orig_copy = torch.Tensor.copy_
    sh.begin_update()
    return sh


def _register(sh, param, name):
    """Mimic the materialize hook: map the param's base storage to its full name."""
    sh._name_by_storage[_storage_ptr(param.data)] = name


def _prime(sh, name, tensor):
    """Mimic ``initialize()``: allocate the whole-param bf16 master for ``name``."""
    sh._shard[name] = tensor.detach().to(torch.bfloat16).clone()


def test_shardshadow_copy_seam_seed_then_delta():
    torch.manual_seed(10)
    sh = _active_shadow()
    try:
        param = torch.zeros(4, 3, dtype=torch.bfloat16)
        _register(sh, param, "w")
        # Master is primed up front (initialize); the seed then overwrites it.
        _prime(sh, "w", torch.zeros(4, 3, dtype=torch.bfloat16))

        # Seed: dense write -> param adopts it and the master records it.
        seed = torch.randn(4, 3, dtype=torch.bfloat16)
        param.copy_(seed)
        assert torch.equal(param, seed)
        assert torch.equal(sh._shard["w"].to(torch.bfloat16), seed)

        # Delta: NaN everywhere except two changed positions -> merge against the master.
        delta = torch.full((4, 3), float("nan"), dtype=torch.bfloat16)
        delta[0, 0] = 9.0
        delta[2, 1] = -3.0
        param.copy_(delta)

        expected = seed.clone()
        expected[0, 0] = 9.0
        expected[2, 1] = -3.0
        assert torch.equal(param, expected)
        # Master now holds the merged values (accumulates across updates).
        assert torch.equal(sh._shard["w"].to(torch.bfloat16), expected)
    finally:
        sh.end_update()
    # The global ``copy_`` is restored after end_update.
    assert torch.Tensor.copy_ is sh._orig_copy


def test_shardshadow_copy_seam_fused_slices_addressed_into_master():
    torch.manual_seed(11)
    sh = _active_shadow()
    try:
        # A qkv-style fused param: three row-slices share one whole-param master.
        param = torch.zeros(6, 4, dtype=torch.bfloat16)
        _register(sh, param, "qkv")
        _prime(sh, "qkv", torch.zeros(6, 4, dtype=torch.bfloat16))
        seeds = [torch.randn(2, 4, dtype=torch.bfloat16) for _ in range(3)]
        for i, s in enumerate(seeds):
            param.narrow(0, 2 * i, 2).copy_(s)

        # The single master is the row-concatenation of the three seeded slices.
        master = sh._shard["qkv"].to(torch.bfloat16)
        assert torch.equal(master, torch.cat(seeds, dim=0))

        # Delta touching only the middle (k) slice must not disturb q/v rows.
        delta = torch.full((2, 4), float("nan"), dtype=torch.bfloat16)
        delta[1, 3] = 7.0
        param.narrow(0, 2, 2).copy_(delta)

        expected_k = seeds[1].clone()
        expected_k[1, 3] = 7.0
        master = sh._shard["qkv"].to(torch.bfloat16)
        assert torch.equal(master[0:2], seeds[0])
        assert torch.equal(master[2:4], expected_k)
        assert torch.equal(master[4:6], seeds[2])
    finally:
        sh.end_update()


def test_shardshadow_copy_seam_ignores_untracked_params():
    sh = _active_shadow()
    try:
        # No registration -> the patch is a pass-through, nothing is shadowed.
        other = torch.zeros(3, dtype=torch.bfloat16)
        other.copy_(torch.ones(3, dtype=torch.bfloat16))
        assert torch.equal(other, torch.ones(3, dtype=torch.bfloat16))
        assert len(sh._shard) == 0
    finally:
        sh.end_update()


def _bf16_model():
    """A tiny nested module whose params are bf16 (mirrors a freshly loaded, pre-quant model).

    ``named_modules`` yields the root (``norm``) and the ``mlp`` submodule (``mlp.weight``), so
    capture must key by the full, prefixed param name.
    """
    root = torch.nn.Module()
    mlp = torch.nn.Module()
    mlp.register_parameter(
        "weight",
        torch.nn.Parameter(torch.randn(4, 3, dtype=torch.bfloat16), requires_grad=False),
    )
    root.add_module("mlp", mlp)
    root.register_parameter(
        "norm",
        torch.nn.Parameter(torch.randn(5, dtype=torch.bfloat16), requires_grad=False),
    )
    return root


def test_prequant_capture_survives_quantization():
    """initialize() adopts the load-time bf16 master, not the (clobbered) post-quant params.

    Models the quantized case: capture the bf16 weights, then let
    ``process_weights_after_loading`` overwrite the live params (here, zero them to stand in
    for a lossy fp8 round-trip). The shadow must reflect the captured bf16, proving it never
    re-reads the now-quantized live param. The stash is keyed by module identity, so capture
    works without any full-name knowledge and ``initialize`` resolves names via the model.
    """
    torch.manual_seed(20)
    model = _bf16_model()
    orig = {
        "mlp.weight": model.mlp.weight.detach().clone(),
        "norm": model.norm.detach().clone(),
    }

    capture_model_prequant_bf16(model)

    with torch.no_grad():
        model.mlp.weight.copy_(torch.zeros_like(model.mlp.weight))
        model.norm.copy_(torch.zeros_like(model.norm))

    sh = ShardShadow()
    sh.initialize(model)
    for name, val in orig.items():
        assert sh.has(name)
        assert torch.equal(sh.shard(name).to(torch.bfloat16), val)


def test_layerwise_capture_survives_param_replacement():
    """Online-quant seam: a per-layer capture survives the param object being *replaced*.

    Mirrors meta-device online fp8, where ``_layerwise_process`` loads bf16 into the layer
    then ``process_weights_after_loading`` swaps ``weight`` for a brand-new fp8 Parameter
    (``replace_parameter``). Module identity is stable across that swap, so the captured
    master still resolves and the live (now-fp8) param is never read.
    """
    torch.manual_seed(21)
    model = _bf16_model()
    orig_w = model.mlp.weight.detach().clone()

    # Capture only the quantized layer (the seam fires per layer, before its quant cast).
    capture_layer_prequant_bf16(model.mlp)

    # process_weights_after_loading: replace the bf16 weight with a fresh fp8 Parameter.
    with torch.no_grad():
        model.mlp.weight = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float8_e4m3fn), requires_grad=False)

    sh = ShardShadow()
    sh.initialize(model)
    # Quantized layer: master is the captured pre-quant bf16, not the fp8 live param.
    assert torch.equal(sh.shard("mlp.weight").to(torch.bfloat16), orig_w)
    # Unquantized param (never captured) still gets a live bf16 snapshot.
    assert torch.equal(sh.shard("norm").to(torch.bfloat16), model.norm.detach())


def test_prequant_capture_is_idempotent():
    """Capture-once per (layer, param): a later call must not overwrite the first master."""
    model = _bf16_model()
    orig_w = model.mlp.weight.detach().clone()

    capture_model_prequant_bf16(model)
    with torch.no_grad():
        model.mlp.weight.copy_(torch.full_like(model.mlp.weight, 9.0))
    capture_model_prequant_bf16(model)  # no-op for already-captured keys

    captured = take_prequant_bf16()
    assert torch.equal(captured[(id(model.mlp), "weight")].to(torch.bfloat16), orig_w)


def test_prequant_capture_skips_non_bf16_params():
    """An already-fp8 param (pre-quantized checkpoint) has no bf16 master and is skipped."""
    root = torch.nn.Module()
    root.register_parameter(
        "w_bf16",
        torch.nn.Parameter(torch.randn(3, dtype=torch.bfloat16), requires_grad=False),
    )
    root.register_parameter(
        "w_fp8",
        torch.nn.Parameter(torch.zeros(3, dtype=torch.float8_e4m3fn), requires_grad=False),
    )

    capture_model_prequant_bf16(root)
    captured = take_prequant_bf16()
    assert (id(root), "w_bf16") in captured
    assert (id(root), "w_fp8") not in captured


def test_take_prequant_bf16_clears_stash():
    """``take`` hands the master off once; a second take is empty."""
    model = _bf16_model()
    capture_model_prequant_bf16(model)
    first = take_prequant_bf16()
    assert set(first) == {(id(model.mlp), "weight"), (id(model), "norm")}
    assert take_prequant_bf16() == {}


def test_initialize_falls_back_to_live_snapshot_without_capture():
    """No load-time capture -> initialize snapshots the live params (correct for a bf16 model)."""
    model = _bf16_model()
    sh = ShardShadow()
    sh.initialize(model)
    assert torch.equal(sh.shard("mlp.weight").to(torch.bfloat16), model.mlp.weight.detach())
    assert torch.equal(sh.shard("norm").to(torch.bfloat16), model.norm.detach())


def test_initialize_skips_uncaptured_fp8_param():
    """A pre-quantized fp8 param with no capture gets no master (the inherent limit)."""
    root = torch.nn.Module()
    root.register_parameter(
        "w_fp8",
        torch.nn.Parameter(torch.zeros(3, dtype=torch.float8_e4m3fn), requires_grad=False),
    )
    sh = ShardShadow()
    sh.initialize(root)
    assert not sh.has("w_fp8")


def test_engine_update_info_validation():
    """The vLLM engine's update info validates list alignment (requires vLLM importable)."""
    pytest.importorskip("vllm")
    from skyrl.backends.skyrl_train.weight_sync.delta_engine import (
        DeltaWeightTransferUpdateInfo,
    )

    with pytest.raises(ValueError, match="must align"):
        DeltaWeightTransferUpdateInfo(
            names=["a", "b"],
            dtype_names=["bfloat16", "bfloat16"],
            shapes=[[2], [2]],
            counts=[1],  # wrong length
            is_seed=False,
        )

    info = DeltaWeightTransferUpdateInfo(
        names=["a"],
        dtype_names=["bfloat16"],
        shapes=[[2]],
        counts=[2],
        is_seed=True,
    )
    assert info.names == ["a"]
    assert info.counts == [2]
    assert info.is_seed is True


class _FakeInferenceClient:
    def __init__(self):
        self.updates = []

    async def update_weights_nccl(self, update_info):
        self.updates.append(update_info)


class _FakeDeltaTransport:
    def __init__(self):
        self.sent = []

    def send(self, payload, *, version, file_index):
        self.sent.append((version, file_index, payload))


@pytest.mark.asyncio
async def test_disk_sender_batches_chunks_by_max_file_size(monkeypatch):
    from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
    from skyrl.backends.skyrl_train.weight_sync.delta_strategy import (
        DeltaInitInfo,
        DeltaWeightTransferSender,
    )

    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    client = _FakeInferenceClient()
    transport = _FakeDeltaTransport()
    sender = DeltaWeightTransferSender(
        DeltaInitInfo(
            override_existing_receiver=False,
            master_addr="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=2,
            group_name="test",
            backend="nccl",
            model_dtype_str="bfloat16",
            transport="disk",
            sync_dir="/tmp/delta",
            max_file_size_in_gb=8 / (2**30),
        ),
        client,
        transport,
    )
    sender._version = 0

    chunks = [
        WeightChunk(names=[name], dtypes=["bfloat16"], shapes=[[2]], tensors=[torch.ones(2, dtype=torch.bfloat16)])
        for name in ("a.weight", "b.weight", "c.weight")
    ]
    await sender._send_chunks_disk(chunks, rank=0)

    assert [update["file_index"] for update in client.updates] == [0, 1]
    assert [update["names"] for update in client.updates] == [["a.weight", "b.weight"], ["c.weight"]]
    assert [(version, file_index) for version, file_index, _ in transport.sent] == [(0, 0), (0, 1)]
    assert [payload.values.numel() for _, _, payload in transport.sent] == [4, 2]


@pytest.mark.asyncio
async def test_disk_sender_uses_int32_positions_for_normal_sparse_chunks(monkeypatch):
    from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
    from skyrl.backends.skyrl_train.weight_sync.delta_strategy import (
        DeltaInitInfo,
        DeltaWeightTransferSender,
    )

    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    client = _FakeInferenceClient()
    transport = _FakeDeltaTransport()
    sender = DeltaWeightTransferSender(
        DeltaInitInfo(
            override_existing_receiver=False,
            master_addr="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=2,
            group_name="test",
            backend="nccl",
            model_dtype_str="bfloat16",
            transport="disk",
            sync_dir="/tmp/delta",
            max_file_size_in_gb=1,
        ),
        client,
        transport,
    )

    base = [
        WeightChunk(
            names=["a.weight"],
            dtypes=["bfloat16"],
            shapes=[[4]],
            tensors=[torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.bfloat16)],
        )
    ]
    await sender._send_chunks_disk(base, rank=0)  # seed the snapshot.
    client.updates.clear()
    transport.sent.clear()

    changed = [
        WeightChunk(
            names=["a.weight"],
            dtypes=["bfloat16"],
            shapes=[[4]],
            tensors=[torch.tensor([0.0, 9.0, 2.0, 3.0], dtype=torch.bfloat16)],
        )
    ]
    await sender._send_chunks_disk(changed, rank=0)

    assert [update["positions_dtype"] for update in client.updates] == ["int32"]
    assert len(transport.sent) == 1
    payload = transport.sent[0][2]
    assert payload.positions is not None
    assert payload.positions.dtype == torch.int32
    assert torch.equal(payload.positions, torch.tensor([1], dtype=torch.int32))


@pytest.mark.asyncio
async def test_disk_sender_honors_int64_positions_transfer_dtype(monkeypatch):
    from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
    from skyrl.backends.skyrl_train.weight_sync.delta_strategy import (
        DeltaInitInfo,
        DeltaWeightTransferSender,
    )

    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    client = _FakeInferenceClient()
    transport = _FakeDeltaTransport()
    sender = DeltaWeightTransferSender(
        DeltaInitInfo(
            override_existing_receiver=False,
            master_addr="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=2,
            group_name="test",
            backend="nccl",
            model_dtype_str="bfloat16",
            transport="disk",
            sync_dir="/tmp/delta",
            positions_transfer_dtype="int64",
        ),
        client,
        transport,
    )

    base = [
        WeightChunk(
            names=["a.weight"],
            dtypes=["bfloat16"],
            shapes=[[4]],
            tensors=[torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.bfloat16)],
        )
    ]
    await sender._send_chunks_disk(base, rank=0)
    client.updates.clear()
    transport.sent.clear()

    changed = [
        WeightChunk(
            names=["a.weight"],
            dtypes=["bfloat16"],
            shapes=[[4]],
            tensors=[torch.tensor([0.0, 9.0, 2.0, 3.0], dtype=torch.bfloat16)],
        )
    ]
    await sender._send_chunks_disk(changed, rank=0)

    assert [update["positions_dtype"] for update in client.updates] == ["int64"]
    assert len(transport.sent) == 1
    payload = transport.sent[0][2]
    assert payload.positions is not None
    assert payload.positions.dtype == torch.int64
    assert torch.equal(payload.positions, torch.tensor([1], dtype=torch.int64))
