// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
// Hash recovery/remapping is adapted from the local snapshots; see UPSTREAM.json.
#include <memory>
#include <mutex>
#include <unordered_map>

#include "cuda_malloc_hash.cuh"
#include "partition_kv.h"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
namespace {
struct Arena {
  uint8_t* base = nullptr;
  int device;
  uint64_t bytes = 64ULL << 30;
  hfuse::RecoveryResult recovery;
  ~Arena() {
    if (base) {
      ffi::CUDADeviceGuard guard(device);
      cudaFree(base);
    }
  }
};
std::mutex arenas_mutex;
std::unordered_map<int64_t, std::unique_ptr<Arena>> arenas;
constexpr uint64_t kMask = flash::PARTITION_MASK_H200;

// Every logical 4 KiB page selects one member of a physical 8 KiB pair.
// Owners reuse the same compact slot numbers without aliasing physical memory.
template <bool GATHER>
__global__ void compact_copy(uint8_t* arena, uint64_t hash_base, uint64_t pool_offset,
                             uint64_t page_stride, uint4* linear, const int* owners,
                             const int* slots, int pages, int vectors_per_page) {
  for (int page = blockIdx.x; page < pages; page += gridDim.x) {
    for (int vec = threadIdx.x; vec < vectors_per_page; vec += blockDim.x) {
      uint64_t logical = pool_offset + uint64_t(slots[page]) * page_stride + uint64_t(vec) * 16;
      auto ptr = owners[page] == 0
                     ? flash::cuda_malloc_partition_ptr<kMask, 0, uint4>(arena, hash_base, logical)
                     : flash::cuda_malloc_partition_ptr<kMask, 1, uint4>(arena, hash_base, logical);
      uint64_t index = uint64_t(page) * vectors_per_page + vec;
      if constexpr (GATHER)
        linear[index] = *ptr;
      else
        *ptr = linear[index];
    }
  }
}
}  // namespace

Array<int64_t> Initialize(int64_t device_id, int64_t reserve_bytes) {
  std::lock_guard<std::mutex> lock(arenas_mutex);
  ffi::CUDADeviceGuard guard(device_id);
  cudaDeviceProp prop{};
  HFUSE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  TVM_FFI_ICHECK(std::string(prop.name).find("H200") != std::string::npos)
      << "This experimental runtime currently supports H200 only";
  auto arena = std::make_unique<Arena>();
  arena->device = device_id;
  size_t free_bytes, total_bytes;
  HFUSE_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  TVM_FFI_ICHECK(reserve_bytes >= 0 && free_bytes >= arena->bytes + uint64_t(reserve_bytes))
      << "Insufficient memory for 64 GiB cudaMalloc arena plus reserve";
  HFUSE_CUDA_CHECK(cudaMalloc(&arena->base, arena->bytes));
  hfuse::ProbeConfig config;
  config.device_id = device_id;
  config.verbose = false;
  arena->recovery =
      hfuse::recover_and_probe(arena->base, arena->bytes, kMask, prop.multiProcessorCount, config);
  TVM_FFI_ICHECK((arena->recovery.hash_base & 8191) == 0)
      << "Recovered base must start on an even 4 KiB page";
  int64_t handle = reinterpret_cast<int64_t>(arena->base);
  Array<int64_t> result{handle, int64_t(arena->bytes), int64_t(arena->recovery.hash_base),
                        int64_t(kMask), int64_t(arena->recovery.threshold_cycles)};
  for (int owner : arena->recovery.sm_partition) result.push_back(owner);
  arenas.emplace(handle, std::move(arena));
  return result;
}

void Release(int64_t handle) {
  std::lock_guard<std::mutex> lock(arenas_mutex);
  arenas.erase(handle);
}

void Copy(int64_t handle, TensorView tensor, TensorView owners, TensorView slots,
          int64_t pool_offset, int64_t page_stride, int64_t slot_capacity, bool gather) {
  std::lock_guard<std::mutex> lock(arenas_mutex);
  TVM_FFI_ICHECK(arenas.count(handle)) << "Arena is closed";
  Arena& arena = *arenas.at(handle);
  ffi::CUDADeviceGuard guard(arena.device);
  TVM_FFI_ICHECK(tensor.device().device_id == arena.device &&
                 tensor.device().device_type == kDLCUDA);
  TVM_FFI_ICHECK(tensor.ndim() == 3 && tensor.stride(2) == 1 &&
                 tensor.stride(1) == tensor.size(2) &&
                 tensor.stride(0) == tensor.size(1) * tensor.size(2));
  int64_t page_bytes = tensor.size(1) * tensor.size(2) * tensor.dtype().bits / 8;
  TVM_FFI_ICHECK(tensor.dtype().bits == 16 && tensor.dtype().lanes == 1);
  TVM_FFI_ICHECK(page_bytes > 0 && page_bytes % 16 == 0 && page_bytes <= page_stride);
  TVM_FFI_ICHECK(pool_offset >= 0 && page_stride > 0 && slot_capacity >= 0 &&
                 pool_offset % 4096 == 0 && page_stride % 4096 == 0 &&
                 uint64_t(pool_offset) + uint64_t(page_stride) * slot_capacity <= arena.bytes / 2);
  for (auto t : {owners, slots}) {
    TVM_FFI_ICHECK(t.ndim() == 1 && t.size(0) == tensor.size(0) && t.stride(0) == 1 &&
                   t.dtype().code == kDLInt && t.dtype().bits == 32 &&
                   t.device().device_type == kDLCUDA && t.device().device_id == arena.device);
  }
  if (!tensor.size(0)) return;
  auto stream = get_stream(tensor.device());
  int blocks = std::min<int64_t>(tensor.size(0), 4096);
  auto kernel = gather ? compact_copy<true> : compact_copy<false>;
  kernel<<<blocks, 256, 0, stream>>>(
      arena.base, arena.recovery.hash_base, pool_offset, page_stride,
      static_cast<uint4*>(tensor.data_ptr()), static_cast<int*>(owners.data_ptr()),
      static_cast<int*>(slots.data_ptr()), tensor.size(0), page_bytes / 16);
  HFUSE_CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize, Initialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(release, Release);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy, Copy);
