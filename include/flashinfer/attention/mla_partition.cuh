// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
#ifndef FLASHINFER_MLA_PARTITION_CUH_
#define FLASHINFER_MLA_PARTITION_CUH_
#include <flashinfer/partition/address.cuh>

#include "mla_params.cuh"
namespace flashinfer {
// Experimental schedule metadata only. KV pointers remain ordinary linear memory.
template <typename Q, typename KV, typename O, typename Id>
struct SM90PartitionMLAParams : MLAParams<Q, KV, O, Id> {
  static constexpr bool PARTITION_SCHEDULE = true;
#ifdef FLASHINFER_MLA_COMPACT_KV
  static constexpr bool COMPACT_KV = true;
  uint8_t* compact_arena;
  uint64_t compact_hash_base;
  uint64_t compact_kpe_offset;
  const int* compact_slots;
#endif
#ifdef FLASHINFER_MLA_NO_SCHEDULE_AUDIT
  static constexpr bool SCHEDULE_AUDIT = false;
#else
  static constexpr bool SCHEDULE_AUDIT = true;
#endif
  const int* sm_partition;
  const int* sm_rank;
  const int* sm_count;
  const Id* task_work;
  const Id* task_q_subtile;
  const Id* owner_indptr;
  int* sm_visits;
  int* task_visits;
  int* task_smid;
  uint32_t logical_grid_x;
};
}  // namespace flashinfer
#endif
