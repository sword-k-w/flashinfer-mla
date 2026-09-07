# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Build the independent H200 cudaMalloc partition runtime."""

from . import env as jit_env
from .core import gen_jit_spec


def gen_partition_runtime_module():
    return gen_jit_spec(
        "h200_partition_runtime_v1",
        [jit_env.FLASHINFER_CSRC_DIR / "partition_runtime" / "runtime.cu"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a"],
    )
