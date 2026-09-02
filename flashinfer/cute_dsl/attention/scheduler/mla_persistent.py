# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLA decode tile scheduler — moved from flashinfer/mla/cute_dsl/mla_helpers.py.

Re-exports the tile scheduler classes and factory functions for use by the
modular MLA decode kernel and its roles.

Also provides host-side utility functions for split-KV computation.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


LOG2_E = 1.4426950408889634074
MAX_SPLITS = 256


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class MLAStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisor = None,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        split_kv_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.split_kv_fdd = split_kv_fdd
        if cutlass.const_expr(problem_shape_b_fdd is None):
            self.problem_shape_b_fdd = cute.fast_divmod_create_divisor(
                problem_shape_b, loc=loc, ip=ip
            )
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor(
                problem_shape_s, loc=loc, ip=ip
            )
        if cutlass.const_expr(split_kv_fdd is None):
            self.split_kv_fdd = cute.fast_divmod_create_divisor(
                split_kv, loc=loc, ip=ip
            )
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.problem_shape_s)
        values += cutlass.extract_mlir_values(self.split_kv)
        values += cutlass.extract_mlir_values(self.problem_shape_b_fdd)
        values += cutlass.extract_mlir_values(self.problem_shape_s_fdd)
        values += cutlass.extract_mlir_values(self.split_kv_fdd)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(
            self.problem_shape_b, (values[0],)
        )
        problem_shape_s = cutlass.new_from_mlir_values(
            self.problem_shape_s, (values[1],)
        )
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[2],))
        problem_shape_b_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_b_fdd, (values[3],)
        )
        problem_shape_s_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_s_fdd, (values[4],)
        )
        split_kv_fdd = cutlass.new_from_mlir_values(self.split_kv_fdd, (values[5],))
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            split_kv,
            problem_shape_b_fdd=problem_shape_b_fdd,
            problem_shape_s_fdd=problem_shape_s_fdd,
            split_kv_fdd=split_kv_fdd,
            loc=self.loc,
        )


class MLAPartitionTileSchedulerParams:
    """Device-local parameters for the experimental split-owner scheduler.

    ``partition`` and the two cluster-rank fields are derived from ``%smid``
    once at kernel entry.  Keeping them as scalars here lets every warp role
    instantiate an identical scheduler without carrying topology tensors
    through the scheduler object.
    """

    def __init__(
        self,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        work_p0: cutlass.Int32,
        partition: cutlass.Int32,
        partition_cluster_rank: cutlass.Int32,
        partition_cluster_count: cutlass.Int32,
        *,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        split_kv_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        self.is_persistent = True
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.work_p0 = work_p0
        self.partition = partition
        self.partition_cluster_rank = partition_cluster_rank
        self.partition_cluster_count = partition_cluster_count
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.split_kv_fdd = split_kv_fdd
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor(
                problem_shape_s, loc=loc, ip=ip
            )
        if cutlass.const_expr(split_kv_fdd is None):
            self.split_kv_fdd = cute.fast_divmod_create_divisor(
                split_kv, loc=loc, ip=ip
            )
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.problem_shape_s)
        values += cutlass.extract_mlir_values(self.split_kv)
        values += cutlass.extract_mlir_values(self.work_p0)
        values += cutlass.extract_mlir_values(self.partition)
        values += cutlass.extract_mlir_values(self.partition_cluster_rank)
        values += cutlass.extract_mlir_values(self.partition_cluster_count)
        values += cutlass.extract_mlir_values(self.problem_shape_s_fdd)
        values += cutlass.extract_mlir_values(self.split_kv_fdd)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(
            self.problem_shape_b, (values[0],)
        )
        problem_shape_s = cutlass.new_from_mlir_values(
            self.problem_shape_s, (values[1],)
        )
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[2],))
        work_p0 = cutlass.new_from_mlir_values(self.work_p0, (values[3],))
        partition = cutlass.new_from_mlir_values(self.partition, (values[4],))
        partition_cluster_rank = cutlass.new_from_mlir_values(
            self.partition_cluster_rank, (values[5],)
        )
        partition_cluster_count = cutlass.new_from_mlir_values(
            self.partition_cluster_count, (values[6],)
        )
        problem_shape_s_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_s_fdd, (values[7],)
        )
        split_kv_fdd = cutlass.new_from_mlir_values(self.split_kv_fdd, (values[8],))
        return MLAPartitionTileSchedulerParams(
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            split_kv,
            work_p0,
            partition,
            partition_cluster_rank,
            partition_cluster_count,
            problem_shape_s_fdd=problem_shape_s_fdd,
            split_kv_fdd=split_kv_fdd,
            loc=self.loc,
        )


def create_mla_partition_tile_scheduler_params(
    base: MLAStaticTileSchedulerParams,
    work_p0: cutlass.Int32,
    partition: cutlass.Int32,
    partition_cluster_rank: cutlass.Int32,
    partition_cluster_count: cutlass.Int32,
) -> MLAPartitionTileSchedulerParams:
    return MLAPartitionTileSchedulerParams(
        base.problem_shape_b,
        base.problem_shape_s,
        base.cluster_shape_mnk,
        base.split_kv,
        work_p0,
        partition,
        partition_cluster_rank,
        partition_cluster_count,
    )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    problem_shape_s: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    return MLAStaticTileSchedulerParams(
        is_persistent, problem_shape_b, problem_shape_s, cluster_shape_mnk, split_kv
    )


class WorkTileInfo:
    def __init__(self, blk_coord: cute.Coord, is_valid: bool):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        return values

    def __new_from_mlir_values__(self, values):
        new_tile_idx = cutlass.new_from_mlir_values(self.blk_coord, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self.is_valid, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


class MLAStaticTileScheduler:
    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        is_valid: bool = True,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        if params.is_persistent:
            self.persistent_blk_layout = cute.make_layout(
                (
                    params.cluster_shape_mnk[0],
                    params.problem_shape_s,
                    params.problem_shape_b,
                    params.split_kv,
                ),
                loc=loc,
                ip=ip,
            )
            self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
            self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        else:
            self.is_valid = is_valid
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: MLAStaticTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        grid_shape = (
            params.cluster_shape_mnk[0],
            params.problem_shape_b * params.problem_shape_s,
            params.split_kv,
        )
        if params.is_persistent:
            return (
                cutlass.min(
                    max_active_clusters * cute.size(params.cluster_shape_mnk),
                    cute.size(grid_shape, loc=loc, ip=ip),
                ),
                1,
                1,
            )
        else:
            return grid_shape

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = (
            self.current_work_linear_idx < self.num_blocks
            if self.params.is_persistent
            else self.is_valid
        )

        if self.params.is_persistent:
            current_work_cluster_batch, cluster_idx = (
                self.current_work_linear_idx // self.params.cluster_shape_mnk[0],
                self.current_work_linear_idx % self.params.cluster_shape_mnk[0],
            )
            current_work_s_batch, s_idx = divmod(
                current_work_cluster_batch, self.params.problem_shape_s_fdd
            )
            current_work_b_batch, b_idx = divmod(
                current_work_s_batch, self.params.problem_shape_b_fdd
            )
            _, split_kv_idx = divmod(current_work_b_batch, self.params.split_kv_fdd)

            blk_coord = (cluster_idx, s_idx, b_idx, split_kv_idx)
        else:
            s_idx, b_idx = divmod(self.blk_coord[1], self.params.problem_shape_b_fdd)
            blk_coord = (self.blk_coord[0], s_idx, b_idx, self.blk_coord[2])

        return WorkTileInfo(blk_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self.params.is_persistent:
            self.current_work_linear_idx += advance_count * self.num_persistent_sm
        else:
            self.is_valid = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 13
        new_params = cutlass.new_from_mlir_values(self.params, values[0:6])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self.current_work_linear_idx, [values[6]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self.blk_coord, values[7:10])
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values[10:])
        return MLAStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


class MLAPartitionTileScheduler:
    """Persistent scheduler whose physical clusters stay with one split owner.

    ``(batch, split_kv)`` work units are flattened in batch-major order. P0
    owns the prefix ``[0, work_p0)`` and P1 owns the remaining suffix. Query
    tiles stay inside their owning work unit. Each physical cluster walks its
    owner's range with a partition-local static stride, with no queue or
    atomics.
    """

    def __init__(
        self,
        params: MLAPartitionTileSchedulerParams,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.grid_shape = grid_shape
        self.current_work_cluster_idx = params.partition_cluster_rank
        total_work_units = params.problem_shape_b * params.split_kv
        self.local_work_unit_count = params.work_p0 + params.partition * (
            total_work_units - 2 * params.work_p0
        )
        self.num_blocks = self.local_work_unit_count * params.problem_shape_s
        self.loc = loc
        self.ip = ip

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = self.current_work_cluster_idx < self.num_blocks
        local_work_unit, s_idx = divmod(
            self.current_work_cluster_idx, self.params.problem_shape_s_fdd
        )
        global_work_unit = local_work_unit + self.params.partition * self.params.work_p0
        b_idx, split_kv_idx = divmod(global_work_unit, self.params.split_kv_fdd)
        cluster_idx = cute.arch.block_idx_in_cluster()
        blk_coord = (cluster_idx, s_idx, b_idx, split_kv_idx)
        return WorkTileInfo(blk_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        self.current_work_cluster_idx += (
            advance_count * self.params.partition_cluster_count
        )

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values += cutlass.extract_mlir_values(self.current_work_cluster_idx)
        values += cutlass.extract_mlir_values(self.grid_shape)
        return values

    def __new_from_mlir_values__(self, values):
        new_params = cutlass.new_from_mlir_values(self.params, values[0:9])
        new_current_work_cluster_idx = cutlass.new_from_mlir_values(
            self.current_work_cluster_idx, [values[9]]
        )
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values[10:])
        scheduler = MLAPartitionTileScheduler(new_params, new_grid_shape)
        scheduler.current_work_cluster_idx = new_current_work_cluster_idx
        return scheduler


def create_mla_static_tile_scheduler(
    params: MLAStaticTileSchedulerParams | MLAPartitionTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> MLAStaticTileScheduler | MLAPartitionTileScheduler:
    if cutlass.const_expr(isinstance(params, MLAPartitionTileSchedulerParams)):
        return MLAPartitionTileScheduler(params, grid_shape)
    return MLAStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


# ---------------------------------------------------------------------------
#  Host-side utilities
# ---------------------------------------------------------------------------


def mla_get_split_kv(
    B: int, S: int, K: int, mma_qk_tiler_mn: tuple, max_active_blocks: int
) -> int:
    """Get split_kv value for MLA kernel (host-side)."""
    max_splits = ceil_div(K, mma_qk_tiler_mn[1])
    blocks_per_batch = max(1, max_active_blocks // B // (S * 2))
    split_heur = min(max_splits, blocks_per_batch)
    k_waves = ceil_div(max_splits, split_heur)
    split_wave_aware = ceil_div(max_splits, k_waves)
    max_split_kv = 32
    return min(split_wave_aware, max_split_kv)


def mla_get_split_kv_simplified(B: int, S: int, max_active_blocks: int) -> int:
    """Simplified split_kv for MLA (host-side, no K dependency)."""
    blocks_per_batch = max(1, max_active_blocks // B // (S * 2))
    max_split_kv = 32
    return min(blocks_per_batch, max_split_kv)


def mla_get_workspace_size(
    H: int, S: int, D: int, B: int, split_kv: int, acc_dtype_width: int
) -> int:
    """Get workspace size in bytes for split-KV MLA decode."""
    if split_kv == 1:
        return 0
    # Decode packs heads into a physical 128-wide MMA-M tile. For H < 128,
    # split-KV partials can still touch the padded head lanes before reduction.
    workspace_heads = max(H, 128)
    return B * workspace_heads * S * split_kv * (D + 1) * acc_dtype_width // 8
