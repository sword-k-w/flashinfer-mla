# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Real H200 validation of the independent runtime and ordinary-KV owner scheduler."""

from pathlib import Path

import pytest
import torch

from benchmarks.sm90_mla_partition import SM90PartitionExperiment
from benchmarks.sm90_mla_separate_merge import make_uniform_case
from flashinfer.jit.partition_runtime import gen_partition_runtime_module
from flashinfer.mla.experimental.partition_runtime import H200PartitionRuntime
from tests.attention.test_sm90_mla_separate_merge import ragged_case, reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(),
    reason="Requires H200 partition probing",
)


@pytest.fixture(scope="module")
def runtime():
    # Probe once; reserve the arena before allocating the test KV caches.
    with H200PartitionRuntime() as value:
        yield value


def test_runtime_is_local_and_topology_is_consistent(runtime):
    root = Path(__file__).resolve().parents[2]
    for source in gen_partition_runtime_module().sources:
        assert Path(source).resolve().is_relative_to(root)
    assert runtime.arena_bytes == 64 << 30
    assert runtime.hash_base % 8192 == 0
    assert (
        runtime.sm_counts_cpu.sum()
        == torch.cuda.get_device_properties(0).multi_processor_count
    )
    assert (runtime.sm_counts_cpu > 0).all()
    for owner in range(2):
        ranks = runtime.sm_rank_cpu[runtime.sm_partition_cpu == owner]
        assert torch.equal(ranks, torch.arange(ranks.numel(), dtype=torch.int32))


@pytest.mark.parametrize("owners", [[0, 1] * 8, [0] * 15 + [1], [1] * 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compact_roundtrip(runtime, owners, dtype):
    # P0/P1 can share compact slot numbers, but their physical pages must not alias.
    layout = runtime.compact_layout(owners)
    ckv = torch.randn(len(owners), 64, 512, device=runtime.device, dtype=dtype)
    kpe = torch.randn(len(owners), 64, 64, device=runtime.device, dtype=dtype)
    for _ in range(2):
        ckv.add_(1)
        kpe.sub_(1)
        layout.scatter(ckv, kpe)
        ck_back, kp_back = layout.gather(dtype)
        torch.testing.assert_close(ck_back, ckv, rtol=0, atol=0)
        torch.testing.assert_close(kp_back, kpe, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,sq,sk",
    [(2, 1, 1024), (64, 1, 32768), (64, 4, 32768), (2, 128, 1024), (2, 1, 1048576)],
)
def test_partition_schedule_matches_fused(runtime, batch, sq, sk):
    baseline = make_uniform_case(batch, sq, sk)
    case = SM90PartitionExperiment(baseline, runtime)
    baseline.original()
    case.float_workspace.fill_(255)
    case.out.fill_(torch.nan)
    case.lse.fill_(torch.nan)
    case.attention()
    case.validate()
    case.merge()
    torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
    torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)
    assert set(case.schedule.page_owners.tolist()) == {0, 1}
    # Every Q sub-tile accessing the same physical KV range has the same owner.
    for entry in case.schedule.range_owners:
        start = entry["kv_indptr"] + entry["kv_start"] // 64
        end = entry["kv_indptr"] + entry["kv_end"] // 64
        assert (case.schedule.page_owners[start:end] == entry["owner"]).all()


def test_mixed_split_and_direct_against_reference(runtime):
    baseline, qi, ki = ragged_case([1, 128], [32768, 1024])
    # Stage 1/2 require identity pages. Restore those after the ragged fixture's shuffle.
    indices = baseline._args[7]
    indices.copy_(
        torch.arange(indices.numel(), dtype=torch.int32, device=indices.device)
    )
    case = SM90PartitionExperiment(baseline, runtime)
    summary = baseline.plan_summary()
    assert summary["split_work_count"] and summary["direct_work_count"]
    baseline.original()
    case.run_checked()
    ref, ref_lse = reference(baseline, qi, ki, [32768, 1024], False, False)
    torch.testing.assert_close(case.out.float(), ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(case.lse, ref_lse, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
    torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("explicit_phases", [False, True])
def test_repeated_static_sm_coverage_and_graph(runtime, return_lse, explicit_phases):
    baseline = make_uniform_case(2, 1, 1024, return_lse=return_lse)
    case = SM90PartitionExperiment(baseline, runtime)
    case.run_checked()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if explicit_phases:
            case.attention()
            case.merge()
        else:
            case.run()
    # Small worklists leave most SMs idle; this stresses early-exit/CTA placement.
    for _ in range(100):
        baseline._args[3].add_(0.001)
        baseline.original()
        case.out.fill_(torch.nan)
        case.float_workspace.fill_(255)
        graph.replay()
        case.validate()
        torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
        if return_lse:
            torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)


def test_validation_detects_missing_work(runtime):
    case = SM90PartitionExperiment(make_uniform_case(2, 1, 1024), runtime)
    case.run_checked()
    case.task_visits[0] = 0
    with pytest.raises(RuntimeError, match="Task coverage failed"):
        case.validate()
    case.run_checked()
    case.sm_visits[0] = 2
    with pytest.raises(RuntimeError, match="Static SM coverage failed"):
        case.validate()


def test_unsupported_inputs_rejected(runtime):
    for baseline in (
        make_uniform_case(2, 1, 1024, causal=True),
        make_uniform_case(2, 1, 1025),
    ):
        with pytest.raises(ValueError):
            SM90PartitionExperiment(baseline, runtime)
    with pytest.raises(ValueError, match="Every physical KV page"):
        runtime.compact_layout([0, -1])


@pytest.mark.parametrize(
    "batch,sq,sk", [(2, 1, 1024), (2, 4, 4096), (2, 128, 1024), (64, 1, 32768)]
)
def test_compact_attention_loads_arena(runtime, batch, sq, sk):
    baseline = make_uniform_case(batch, sq, sk)
    baseline.original()
    case = SM90PartitionExperiment(baseline, runtime, compact=True)
    # Prove attention loads the arena, not a hidden fallback to the source tensors.
    baseline._args[5].fill_(torch.nan)
    baseline._args[6].fill_(torch.nan)
    case.float_workspace.fill_(255)
    case.run_checked()
    torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
    torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)
    case.set_audit(False)
    case.out.fill_(torch.nan)
    case.run()
    torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
    torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="compiled out"):
        case.validate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.attention()
        case.merge()
    for _ in range(5):
        case.float_workspace.fill_(255)
        case.out.fill_(torch.nan)
        graph.replay()
        torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
        torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)


def test_noaudit_linear_schedule(runtime):
    baseline = make_uniform_case(2, 1, 1024)
    baseline.original()
    case = SM90PartitionExperiment(baseline, runtime, audit=False)
    # Timing specialization must neither update nor reset the diagnostic buffers.
    case.sm_visits.fill_(17)
    case.task_visits.fill_(19)
    for _ in range(20):
        case.run()
        torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
        torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)
    assert (case.sm_visits == 17).all()
    assert (case.task_visits == 19).all()
