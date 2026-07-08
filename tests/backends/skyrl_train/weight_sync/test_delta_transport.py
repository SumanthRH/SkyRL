"""CPU unit tests for the delta weight-sync transport layer + integrity checksum.

These exercise the transport-agnostic pieces added for disk-based delta sync, all runnable
on CPU without vLLM, NCCL, or a distributed env:

* :func:`delta_checksum` / :func:`verify_delta_checksum` -- the CRC32 integrity guard;
* :func:`build_delta_manifest` / :func:`iter_delta_params` -- the unified per-file manifest;
* :class:`DiskDeltaTransport` -- safetensors round-trip, versioned layout, retention cleanup;
* :class:`NcclDeltaTransport` -- the ``total == 0`` short-circuit (no broadcast) and the
  non-empty broadcast path against a fake group (CPU-only; the real GPU collective path is
  covered by the GPU weight-sync tests).
"""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.delta_transport import (
    DeltaPayload,
    DiskDeltaTransport,
    NcclDeltaTransport,
)
from skyrl.backends.skyrl_train.weight_sync.delta_utils import (
    POSITION_DTYPE,
    build_delta_manifest,
    delta_checksum,
    iter_delta_params,
    verify_delta_checksum,
)

# --- checksum --------------------------------------------------------------------------


def test_checksum_roundtrip_delta_and_seed():
    positions = torch.tensor([1, 5, 9], dtype=POSITION_DTYPE)
    values = torch.tensor([0.5, -1.0, 2.0], dtype=torch.bfloat16)

    # Delta: positions + values. Recompute over the same bytes -> matches.
    crc = delta_checksum(positions, values)
    verify_delta_checksum(crc, positions.clone(), values.clone())

    # Seed: positions=None (dense). A different (positions=None) checksum than the delta.
    seed_crc = delta_checksum(None, values)
    verify_delta_checksum(seed_crc, None, values.clone())
    assert seed_crc != crc


def test_checksum_detects_corruption():
    positions = torch.tensor([0, 2], dtype=POSITION_DTYPE)
    values = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    crc = delta_checksum(positions, values)

    corrupted = values.clone()
    corrupted[0] = 9.0
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_delta_checksum(crc, positions, corrupted)


def test_checksum_empty_payload_is_zero():
    empty_pos = torch.empty(0, dtype=POSITION_DTYPE)
    empty_val = torch.empty(0, dtype=torch.bfloat16)
    assert delta_checksum(empty_pos, empty_val) == 0
    verify_delta_checksum(0, empty_pos, empty_val)


# --- unified manifest ------------------------------------------------------------------


def test_build_manifest_and_iter_params():
    manifest = build_delta_manifest(
        names=["a.weight", "b.weight"],
        dtype_names=["bfloat16", "bfloat16"],
        shapes=[[4, 3], [5]],
        counts=[2, 0],
        is_seed=False,
        checksum=1234,
        version=7,
        file_index=3,
        positions_dtype="int32",
    )
    assert manifest["names"] == ["a.weight", "b.weight"]
    assert manifest["counts"] == [2, 0]
    assert manifest["is_seed"] is False
    assert manifest["checksum"] == 1234
    assert manifest["version"] == 7
    assert manifest["file_index"] == 3
    assert manifest["positions_dtype"] == "int32"

    params = list(iter_delta_params(manifest["names"], manifest["dtype_names"], manifest["shapes"], manifest["counts"]))
    assert [p.name for p in params] == ["a.weight", "b.weight"]
    assert [p.count for p in params] == [2, 0]
    assert params[0].shape == [4, 3]


# --- disk transport --------------------------------------------------------------------


def _payload_seed(values):
    return DeltaPayload(values=values, positions=None)


def _payload_delta(positions, values):
    return DeltaPayload(values=values, positions=positions)


def test_disk_transport_requires_sync_dir():
    with pytest.raises(ValueError, match="non-empty sync_dir"):
        DiskDeltaTransport("")


def test_disk_transport_seed_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    transport = DiskDeltaTransport(str(tmp_path))
    values = torch.randn(12, dtype=torch.bfloat16)

    transport.begin_sync(0)
    transport.send(_payload_seed(values), version=0, file_index=0)

    values_cpu, positions_cpu = transport.receive(total=12, is_seed=True, version=0, file_index=0)
    assert positions_cpu is None
    assert torch.equal(values_cpu, values)


def test_disk_transport_delta_roundtrip_with_checksum(tmp_path):
    pytest.importorskip("safetensors")
    transport = DiskDeltaTransport(str(tmp_path))
    positions = torch.tensor([0, 3, 7], dtype=torch.int32)
    values = torch.tensor([1.0, -2.0, 0.25], dtype=torch.bfloat16)
    crc = delta_checksum(positions, values)

    transport.begin_sync(1)
    transport.send(_payload_delta(positions, values), version=1, file_index=2)

    values_cpu, positions_cpu = transport.receive(
        total=3,
        is_seed=False,
        version=1,
        file_index=2,
        positions_dtype=torch.int32,
    )
    assert torch.equal(positions_cpu, positions)
    assert positions_cpu.dtype == torch.int32
    assert torch.equal(values_cpu, values)
    # End-to-end integrity: the bytes that landed on disk verify against the sender's CRC.
    verify_delta_checksum(crc, positions_cpu, values_cpu)


def test_disk_transport_empty_chunk_writes_no_file(tmp_path):
    pytest.importorskip("safetensors")
    transport = DiskDeltaTransport(str(tmp_path))
    transport.begin_sync(0)
    # No changed elements -> nothing on the wire, no file written.
    empty_payload = DeltaPayload(
        values=torch.empty(0, dtype=torch.bfloat16),
        positions=torch.empty(0, dtype=POSITION_DTYPE),
    )
    transport.send(empty_payload, version=0, file_index=0)
    assert not (tmp_path / "weight_v000000" / "file_00000.safetensors").exists()

    values_cpu, positions_cpu = transport.receive(
        total=0,
        is_seed=False,
        version=0,
        file_index=0,
        positions_dtype=torch.int32,
    )
    assert values_cpu.numel() == 0
    assert positions_cpu is not None and positions_cpu.numel() == 0
    assert positions_cpu.dtype == torch.int32


def test_disk_transport_is_append_only(tmp_path):
    """Without a retention limit, each sync writes a fresh dir and leaves prior files alone."""
    pytest.importorskip("safetensors")
    transport = DiskDeltaTransport(str(tmp_path))
    values = torch.randn(4, dtype=torch.bfloat16)

    for v in range(3):
        transport.begin_sync(v)
        transport.send(_payload_seed(values), version=v, file_index=0)
        transport.end_sync(v)  # inherited base no-op; kept for the lifecycle contract

    # All three versions remain on disk when file retention is disabled.
    for v in range(3):
        assert (tmp_path / f"weight_v{v:06d}").is_dir()


def test_disk_transport_cleanup_keeps_recent_files(tmp_path):
    pytest.importorskip("safetensors")
    transport = DiskDeltaTransport(str(tmp_path), max_files_to_keep=1)
    values = torch.randn(4, dtype=torch.bfloat16)

    transport.begin_sync(0)
    for file_index in range(3):
        transport.send(_payload_seed(values + file_index), version=0, file_index=file_index)
    transport.end_sync(0)

    version_dir = tmp_path / "weight_v000000"
    assert not (version_dir / "file_00000.safetensors").exists()
    assert not (version_dir / "file_00001.safetensors").exists()
    assert (version_dir / "file_00002.safetensors").exists()


# --- nccl transport (fake group, CPU) --------------------------------------------------


class _FakeGroup:
    """Records every broadcast so tests can assert the ``total == 0`` guard skips the wire."""

    def __init__(self):
        self.broadcasts = []

    def broadcast(self, tensor, src, stream):
        self.broadcasts.append(tensor.clone())


def test_nccl_transport_empty_send_skips_broadcast():
    group = _FakeGroup()
    transport = NcclDeltaTransport(group, torch.device("cpu"))
    empty = DeltaPayload(
        values=torch.empty(0, dtype=torch.bfloat16),
        positions=torch.empty(0, dtype=POSITION_DTYPE),
    )
    transport.send(empty, version=0, file_index=0)
    assert group.broadcasts == []


def test_nccl_transport_total_zero_receive_skips_broadcast():
    group = _FakeGroup()
    transport = NcclDeltaTransport(group, torch.device("cpu"))

    # Seed: nothing changed -> no broadcast, empty values, no positions.
    values_cpu, positions_cpu = transport.receive(total=0, is_seed=True, version=0, file_index=0)
    assert values_cpu.numel() == 0 and positions_cpu is None

    # Delta: nothing changed -> no broadcast, empty values + empty positions.
    values_cpu, positions_cpu = transport.receive(total=0, is_seed=False, version=0, file_index=0)
    assert values_cpu.numel() == 0
    assert positions_cpu is not None and positions_cpu.numel() == 0
    assert group.broadcasts == []


def test_nccl_transport_nonempty_send_broadcasts(monkeypatch):
    # ``_broadcast`` asks for the current CUDA stream; stub it so the path runs on CPU.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: None)
    group = _FakeGroup()
    transport = NcclDeltaTransport(group, torch.device("cpu"))

    positions = torch.tensor([0, 2], dtype=POSITION_DTYPE)
    values = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
    transport.send(DeltaPayload(values=values, positions=positions), version=0, file_index=0)
    # A delta broadcasts both values and positions.
    assert len(group.broadcasts) == 2
    assert torch.equal(group.broadcasts[0], values)
    assert torch.equal(group.broadcasts[1], positions)
