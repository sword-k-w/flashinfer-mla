# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Stage 1/2: local H200 runtime and static owner scheduling with ordinary KV.

Run: python -m benchmarks.sm90_mla_partition --output report.json
"""

import argparse
import functools
import json
from pathlib import Path

import torch

from benchmarks.sm90_mla_separate_merge import make_uniform_case
from flashinfer.jit.attention import gen_batch_mla_module
from flashinfer.mla.experimental.partition_runtime import H200PartitionRuntime
from flashinfer.mla.experimental.partition_schedule import build_owner_schedule


@functools.cache
def _partition_module(dtype, compact=False, audit=True):
    return gen_batch_mla_module(
        "fa3",
        dtype,
        dtype,
        dtype,
        torch.int32,
        512,
        64,
        False,
        separate_merge=True,
        partition_schedule=True,
        compact_kv=compact,
        schedule_audit=audit,
    ).build_and_load()


class SM90PartitionExperiment:
    """Wrap the separate-merge baseline with an audited SM-local static schedule.

    compact=True loads the local cudaMalloc arena; otherwise KV is ordinary.
    Call validate() after asynchronous launches
    or graph replays; run_checked() performs this check for an eager full run.
    This experimental static mapping requires exclusive GPU execution and one
    persistent CTA per SM; coverage is explicitly checked, not assumed from grid IDs.
    """

    def __init__(self, baseline, runtime, *, compact=False, audit=True, schedule=None):
        self.baseline, self.runtime = baseline, runtime
        qn, _, ckv, _, indices = baseline._args[3:8]
        if (
            qn.device != runtime.device
            or qn.dtype != torch.bfloat16
            or baseline._args[10] != 0
        ):
            raise ValueError(
                "First-stage schedule requires noncausal BF16 on the runtime device"
            )
        self.compact, self.audit = compact, audit
        self.schedule = schedule or build_owner_schedule(
            baseline.plan,
            baseline.int_workspace,
            indices,
            ckv.shape[0],
            qn.shape[1],
            ckv.shape[1],
            runtime.sm_counts_cpu,
        )
        self.module = _partition_module(qn.dtype, compact, audit)
        self.float_workspace = torch.empty_like(baseline.float_workspace)
        self.out = torch.empty_like(baseline.out)
        self.lse = torch.empty_like(baseline.lse) if baseline.lse is not None else None
        args = list(baseline._args)
        args[0], args[8], args[9] = self.float_workspace, self.out, self.lse
        self._args = tuple(args)
        self.sm_visits = torch.empty_like(runtime.sm_partition)
        self.task_visits = torch.empty_like(
            self.schedule.task_work, device=runtime.device
        )
        self.task_smid = torch.empty_like(self.task_visits)
        self._schedule_args = (
            runtime.sm_partition,
            runtime.sm_rank,
            runtime.sm_counts,
            self.schedule.task_work.to(runtime.device),
            self.schedule.task_q_subtile.to(runtime.device),
            self.schedule.owner_indptr.to(runtime.device),
            self.sm_visits,
            self.task_visits,
            self.task_smid,
        )
        self.layout = None
        self._compact_args = ()
        if compact:
            self.layout = runtime.compact_layout(self.schedule.page_owners)
            self.layout.scatter(*baseline._args[5:7])
            self._compact_args = (
                runtime.handle,
                runtime.hash_base,
                self.layout.kpe_offset,
                self.layout.slots,
            )
        expected = []
        for owner in range(2):
            ids = torch.where(runtime.sm_partition_cpu == owner)[0].tolist()
            begin, end = map(int, self.schedule.owner_indptr[owner : owner + 2])
            expected.extend(ids[i % len(ids)] for i in range(end - begin))
        self.expected_smid = torch.tensor(expected, dtype=torch.int32)

    def attention(self):
        self.module.run(*self._args, 1, *self._schedule_args, *self._compact_args)

    def merge(self):
        self.module.run(*self._args, 2, *self._schedule_args, *self._compact_args)
        return self.out, self.lse

    def run(self):
        self.module.run(*self._args, 0, *self._schedule_args, *self._compact_args)
        return self.out, self.lse

    def set_audit(self, enabled):
        """Switch compiled specialization before capture/timing (same data/plan)."""
        self.module = _partition_module(self._args[3].dtype, self.compact, enabled)
        self.audit = enabled

    def validate(self):
        if not self.audit:
            raise RuntimeError(
                "Schedule audit is compiled out; no counters to validate"
            )
        sm_visits = self.sm_visits.cpu()
        visits = self.task_visits.cpu()
        smids = self.task_smid.cpu()
        if not (sm_visits == 1).all():
            raise RuntimeError(f"Static SM coverage failed: {sm_visits.tolist()}")
        if not (visits == 1).all():
            raise RuntimeError(f"Task coverage failed: {visits.tolist()}")
        if not torch.equal(smids, self.expected_smid):
            raise RuntimeError("A task ran on an unexpected owner/local SM rank")
        if not torch.equal(
            self.runtime.sm_partition_cpu[smids.long()], self.schedule.task_owners
        ):
            raise RuntimeError("Task partition differs from data owner")
        return dict(
            sms=int(sm_visits.numel()),
            tasks=int(visits.numel()),
            duplicate_tasks=0,
            missing_tasks=0,
            wrong_partition_tasks=0,
            owner_task_counts=self.schedule.owner_indptr.diff().tolist(),
            owner_cost=self.schedule.owner_cost,
            owner_page_counts=torch.bincount(
                self.schedule.page_owners.long(), minlength=2
            ).tolist(),
        )

    def run_checked(self):
        result = self.run()
        self.validate()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="B,Sq,Sk")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(42)
    records = []
    with H200PartitionRuntime() as runtime:
        print("runtime", runtime.sm_counts_cpu.tolist(), flush=True)
        for value in args.case or [
            "2,1,1024",
            "64,1,32768",
            "64,4,32768",
            "2,128,1024",
            "2,1,1048576",
        ]:
            b, sq, sk = map(int, value.split(","))
            baseline = make_uniform_case(b, sq, sk)
            case = SM90PartitionExperiment(baseline, runtime)
            baseline.original()
            case.float_workspace.fill_(255)
            case.out.fill_(torch.nan)
            case.run_checked()
            torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
            torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)
            layout = runtime.compact_layout(case.schedule.page_owners)
            ckv, kpe = baseline._args[5:7]
            layout.scatter(ckv, kpe)
            ck_back, kp_back = layout.gather()
            torch.testing.assert_close(ck_back, ckv, rtol=0, atol=0)
            torch.testing.assert_close(kp_back, kpe, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                case.attention()
                case.merge()
            for _ in range(args.replays):
                graph.replay()
                case.validate()
            torch.testing.assert_close(case.out, baseline.original_out, rtol=0, atol=0)
            torch.testing.assert_close(case.lse, baseline.original_lse, rtol=0, atol=0)
            rec = dict(
                batch=b,
                sq=sq,
                sk=sk,
                **case.validate(),
                compact_roundtrip="bitwise equal",
                output_lse="bitwise equal to fused",
                graph_replays=args.replays,
            )
            records.append(rec)
            print(json.dumps(rec), flush=True)
            del graph, case, baseline, layout, ck_back, kp_back, ckv, kpe
        report = dict(
            gpu=torch.cuda.get_device_name(),
            arena_bytes=runtime.arena_bytes,
            hash_base=runtime.hash_base,
            mask=runtime.mask,
            sm_partition=runtime.sm_partition_cpu.tolist(),
            sm_rank=runtime.sm_rank_cpu.tolist(),
            sm_counts=runtime.sm_counts_cpu.tolist(),
            cases=records,
            attention_kv="ordinary linear tensors; compact loading is stage 3",
        )
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
