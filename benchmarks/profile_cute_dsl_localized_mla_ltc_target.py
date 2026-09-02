#!/usr/bin/env python3
"""Launch one standard or localized MLA decode target under Nsight Compute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl.attention.experimental.localized_mla import (
    LocalizedMLAKVCache,
)
from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
    _get_split_kv_and_workspace_size,
)
from flashinfer.cute_dsl.utils import get_num_sm
from localized_mla_benchmark import (
    DTYPE,
    HEADS,
    LATENT_DIM,
    PAGE_SIZE,
    ROPE_DIM,
    compiled_kernel,
    deepseek_v3_effective_softmax_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seqlen-k", type=int, required=True)
    parser.add_argument("--mode", choices=("standard", "localized"), required=True)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if args.batch < 2:
        parser.error("--batch must be >= 2")
    if args.seqlen_k <= 0 or args.seqlen_k % PAGE_SIZE:
        parser.error(f"--seqlen-k must be a positive multiple of {PAGE_SIZE}")
    return args


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    pages_per_batch = args.seqlen_k // PAGE_SIZE
    num_pages = args.batch * pages_per_batch
    softmax_scale = deepseek_v3_effective_softmax_scale()
    cache = None

    try:
        # NCU profiles the layouts in isolated processes, matching the prior
        # vllm-fa LTC experiment and leaving profiler headroom at capacity.
        if args.mode == "localized":
            cache = LocalizedMLAKVCache(
                args.batch,
                args.seqlen_k,
                page_size=PAGE_SIZE,
                dtype=DTYPE,
                device=device,
            )
            c_latent_p0 = cache.kv_p0[..., :LATENT_DIM]
            c_rope_p0 = cache.kv_p0[..., LATENT_DIM:]
            c_latent_p1 = cache.kv_p1[..., :LATENT_DIM]
            c_rope_p1 = cache.kv_p1[..., LATENT_DIM:]
            page_table = cache.page_table
            standard_kv = None
        else:
            cache = None
            standard_kv = torch.empty(
                num_pages,
                PAGE_SIZE,
                LATENT_DIM + ROPE_DIM,
                dtype=DTYPE,
                device=device,
            )
            c_latent_p0 = standard_kv[..., :LATENT_DIM]
            c_rope_p0 = standard_kv[..., LATENT_DIM:]
            c_latent_p1 = None
            c_rope_p1 = None
            page_table = torch.arange(
                num_pages, dtype=torch.int32, device=device
            ).reshape(args.batch, pages_per_batch)

        query = torch.empty(
            args.batch,
            1,
            HEADS,
            LATENT_DIM + ROPE_DIM,
            dtype=DTYPE,
            device=device,
        )
        q_latent = query[..., :LATENT_DIM]
        q_rope = query[..., LATENT_DIM:]
        seq_lens = torch.full(
            (args.batch,), args.seqlen_k, dtype=torch.int32, device=device
        )
        split_kv, workspace_size = _get_split_kv_and_workspace_size(
            args.batch,
            1,
            HEADS,
            LATENT_DIM,
            get_num_sm(device),
        )
        if cache is not None and cache.split_kv != split_kv:
            raise RuntimeError("localized cache and kernel split_kv disagree")
        workspace = (
            None
            if workspace_size == 0
            else torch.empty(workspace_size, dtype=torch.int8, device=device)
        )
        out = torch.empty(args.batch, 1, HEADS, LATENT_DIM, dtype=DTYPE, device=device)
        lse = torch.empty(args.batch, 1, HEADS, dtype=torch.float32, device=device)
        kernel = compiled_kernel(
            partition_aware=args.mode == "localized",
            workspace_size_zero=workspace_size == 0,
        )

        def target_call():
            return kernel(
                q_latent,
                q_rope,
                c_latent_p0,
                c_rope_p0,
                c_latent_p1,
                c_rope_p1,
                page_table,
                out,
                lse,
                workspace,
                Int32(split_kv),
                seq_lens,
                None,
                None if cache is None else cache.sm_partition_map,
                None if cache is None else cache.sm_cluster_rank,
                None if cache is None else cache.partition_cluster_count,
                Int32(0 if cache is None else cache.work_p0),
                Float32(softmax_scale),
                Float32(1.0),
                None,
            )

        metadata = {
            "mode": args.mode,
            "batch_size": args.batch,
            "seqlen_q": 1,
            "seqlen_k": args.seqlen_k,
            "split_kv": split_kv,
            "warmup_launches": args.warmups,
            "profiled_launches": 1,
            "total_sm_count": properties.multi_processor_count,
            "device": properties.name,
            "owner_work_counts": (
                None if cache is None else [cache.work_p0, cache.work_p1]
            ),
            "owner_page_counts": (
                None if cache is None else list(cache.owner_page_counts)
            ),
            "resident_partition_clusters": (
                None if cache is None else cache.partition_cluster_count.cpu().tolist()
            ),
            "localized_mapped_bytes": (
                None if cache is None else list(cache.mapped_bytes)
            ),
        }
        print("TARGET_METADATA " + json.dumps(metadata, sort_keys=True), flush=True)
        for _ in range(args.warmups):
            target_call()
        torch.cuda.synchronize(device)

        # NCU is invoked with --profile-from-start off.  This restricts
        # collection to the one decode launch below, independent of JIT/setup
        # kernel counts and without relying on a fragile mangled-name regex.
        torch.cuda.cudart().cudaProfilerStart()
        target_call()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    finally:
        if cache is not None:
            c_latent_p0 = None
            c_rope_p0 = None
            c_latent_p1 = None
            c_rope_p1 = None
            cache.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
