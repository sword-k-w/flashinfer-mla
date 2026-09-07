/******************************************************************************
 * Partition-aware KV cache support for Flash Attention.
 *
 * Provides DRAM partition address remapping for a cudaMalloc arena whose
 * hash-equivalent physical base has been recovered and validated.
 *
 * Adapted from hfuse/common/partition.cuh — uses the same O(1) algorithm.
 * Both Mask and PartitionId are template parameters so the compiler can:
 *   - Fold (Mask >> 12) into a constant PAGE_MASK
 *   - Fold the XOR with PartitionId
 *   - Potentially strength-reduce the __popcll with a known bitmask
 *
 * Callers branch once on (gpu_type, my_partition) and invoke the appropriate
 * instantiation. B200 and B300 share one Blackwell hash-equivalent mask and
 * therefore one pair of template instantiations.
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash {

// Known GPU partition masks
static constexpr uint64_t PARTITION_MASK_H100 = 0x00000013a78b3000ULL;
static constexpr uint64_t PARTITION_MASK_H200 = 0x0000001cbd6ab000ULL;
static constexpr uint64_t PARTITION_MASK_BLACKWELL = 0x00000003ad3ef000ULL;
static constexpr uint64_t PARTITION_MASK_B200 = PARTITION_MASK_BLACKWELL;
// The matching hash base is recovered and validated independently for every
// B300 cudaMalloc arena; this is an arena-relative hash-equivalent mask.
static constexpr uint64_t PARTITION_MASK_B300 = PARTITION_MASK_BLACKWELL;

template <uint64_t Mask, int PartitionId>
__host__ __device__ __forceinline__
uint64_t remap_partition_addr(uint64_t start_addr, uint64_t ori_address) {
    static_assert(PartitionId == 0 || PartitionId == 1, "PartitionId must be 0 or 1");
    static_assert((Mask & 0xFFF) == 0, "lowest hash bit must be >= bit 12");
    static_assert((Mask & 0x1000) != 0, "partition mask must include bit 12");
    constexpr uint64_t PAGE_MASK = Mask >> 12;
    uint64_t offset     = ori_address - start_addr;
    uint64_t page_j     = offset >> 12;
    uint64_t byte_off   = offset & 0xFFF;
    uint64_t start_page = start_addr >> 12;
    uint64_t even_page  = start_page + 2 * page_j;
#if defined(__CUDA_ARCH__)
    int delta = (__popcll(even_page & PAGE_MASK) & 1) ^ PartitionId;
#else
    int delta = __builtin_parityll(even_page & PAGE_MASK) ^ PartitionId;
#endif
    return start_addr + (page_j << 13) + ((uint64_t)delta << 12) + byte_off;
}

// Convert a logical arena byte offset to a CUDA virtual byte offset. Arena
// initialization guarantees that the recovered base starts on an even 4 KiB
// page, so every remap pair begins at logical offset zero.
template <uint64_t Mask, int PartitionId>
__host__ __device__ __forceinline__
uint64_t cuda_malloc_partition_offset(uint64_t hash_base,
                                      uint64_t logical_byte_offset) {
    uint64_t remapped = remap_partition_addr<Mask, PartitionId>(
        hash_base, hash_base + logical_byte_offset);
    return remapped - hash_base;
}

template <uint64_t Mask, int PartitionId, typename Element>
__device__ __forceinline__
Element* cuda_malloc_partition_ptr(uint8_t* arena_base, uint64_t hash_base,
                                   uint64_t logical_byte_offset) {
    return reinterpret_cast<Element*>(
        arena_base + cuda_malloc_partition_offset<Mask, PartitionId>(
                         hash_base, logical_byte_offset));
}

} // namespace flash
