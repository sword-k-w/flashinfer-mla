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
from triton import runtime as triton_runtime

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
    parser.add_argument("--seqlen-q", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--seqlen-k", type=int, required=True)
    parser.add_argument("--mode", choices=("standard", "localized"), required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--data-initialization",
        choices=("empty", "random"),
        default="empty",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="Clear Triton's benchmark cache before every attention launch.",
    )
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
    standard_kv = None
    initialization_source = None

    try:
        # NCU profiles the layouts in isolated processes, matching the prior
        # vllm-fa LTC experiment and leaving profiler headroom at capacity.
        if args.mode == "localized":
            cache = LocalizedMLAKVCache(
                args.batch,
                args.seqlen_k,
                seq_len_q=args.seqlen_q,
                page_size=PAGE_SIZE,
                dtype=DTYPE,
                device=device,
            )
            c_latent_p0 = cache.kv_p0[..., :LATENT_DIM]
            c_rope_p0 = cache.kv_p0[..., LATENT_DIM:]
            c_latent_p1 = cache.kv_p1[..., :LATENT_DIM]
            c_rope_p1 = cache.kv_p1[..., LATENT_DIM:]
            page_table = cache.page_table
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

        if args.data_initialization == "random":
            torch.manual_seed(args.seed)
            if cache is None:
                standard_kv.normal_()
            else:
                # Generate the same logical KV values as the standard process,
                # scatter them into owner-local storage, then release the large
                # temporary before NCU creates any kernel-replay backing state.
                initialization_source = torch.empty(
                    num_pages,
                    PAGE_SIZE,
                    LATENT_DIM + ROPE_DIM,
                    dtype=DTYPE,
                    device=device,
                )
                initialization_source.normal_()
                cache.scatter_from(initialization_source)

        query = torch.empty(
            args.batch,
            args.seqlen_q,
            HEADS,
            LATENT_DIM + ROPE_DIM,
            dtype=DTYPE,
            device=device,
        )
        if args.data_initialization == "random":
            query.normal_()
            if initialization_source is not None:
                torch.cuda.synchronize(device)
                initialization_source = None
                torch.cuda.empty_cache()
        q_latent = query[..., :LATENT_DIM]
        q_rope = query[..., LATENT_DIM:]
        seq_lens = torch.full(
            (args.batch,), args.seqlen_k, dtype=torch.int32, device=device
        )
        split_kv, workspace_size = _get_split_kv_and_workspace_size(
            args.batch,
            args.seqlen_q,
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
        out = torch.empty(
            args.batch,
            args.seqlen_q,
            HEADS,
            LATENT_DIM,
            dtype=DTYPE,
            device=device,
        )
        lse = torch.empty(
            args.batch,
            args.seqlen_q,
            HEADS,
            dtype=torch.float32,
            device=device,
        )
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

        l2_flush_cache = (
            triton_runtime.driver.active.get_empty_cache_for_benchmark()
            if args.cold_l2
            else None
        )

        def warmup_call():
            if l2_flush_cache is not None:
                triton_runtime.driver.active.clear_cache(l2_flush_cache)
            target_call()

        metadata = {
            "mode": args.mode,
            "batch_size": args.batch,
            "seqlen_q": args.seqlen_q,
            "seqlen_k": args.seqlen_k,
            "split_kv": split_kv,
            "warmup_launches": args.warmups,
            "profiled_launches": 1,
            "data_initialization": args.data_initialization,
            "data_seed": args.seed if args.data_initialization == "random" else None,
            "l2_cache_policy": (
                "triton_clear_cache_before_each_attention_launch"
                if args.cold_l2
                else "unchanged_after_warmups"
            ),
            "total_sm_count": properties.multi_processor_count,
            "device": properties.name,
            "partition_sm_counts": (
                None
                if cache is None
                else [
                    int((cache.sm_partition_map == 0).sum().item()),
                    int((cache.sm_partition_map == 1).sum().item()),
                ]
            ),
            "owner_work_counts": (
                None if cache is None else [cache.work_p0, cache.work_p1]
            ),
            "owner_tile_counts": (
                None
                if cache is None
                else [
                    cache.work_p0 * args.seqlen_q,
                    cache.work_p1 * args.seqlen_q,
                ]
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
            warmup_call()
        torch.cuda.synchronize(device)

        # NCU is invoked with --profile-from-start off.  This restricts
        # collection to the one decode launch below, independent of JIT/setup
        # kernel counts and without relying on a fragile mangled-name regex.
        # Clear and synchronize before enabling collection so the cache-clear
        # operation itself cannot consume --launch-count=1.
        if l2_flush_cache is not None:
            triton_runtime.driver.active.clear_cache(l2_flush_cache)
            torch.cuda.synchronize(device)
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
            initialization_source = None
            cache.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
