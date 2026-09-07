#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hfuse {

constexpr int kPartitionPageBits = 12;
constexpr uint64_t kPartitionPageSize = 1ULL << kPartitionPageBits;
constexpr uint64_t kProbeLineSize = 128;
constexpr int kLatencyTrials = 15;

inline void cuda_check(cudaError_t result, const char* expression,
                       const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA failure at ") + file + ":" + std::to_string(line) +
        " for " + expression + ": " + cudaGetErrorString(result));
}

#define HFUSE_CUDA_CHECK(expression) \
    ::hfuse::cuda_check((expression), #expression, __FILE__, __LINE__)

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error("partition-aware cudaMalloc: " + message);
}

inline int highest_set_bit(uint64_t value) {
    return value == 0 ? -1 : 63 - __builtin_clzll(value);
}

inline uint64_t low_bits_mask(int bits) {
    if (bits <= 0) {
        return 0;
    }
    if (bits >= 64) {
        return ~0ULL;
    }
    return (1ULL << bits) - 1ULL;
}

inline int partition_hash(uint64_t mask, uint64_t address) {
    return __builtin_parityll(address & mask);
}

inline size_t required_allocation_bytes(uint64_t mask) {
    int highest_bit = highest_set_bit(mask);
    if (highest_bit < 0 || highest_bit >= static_cast<int>(sizeof(size_t) * 8)) {
        fail("partition mask cannot be represented as an allocation size");
    }
    return size_t{1} << highest_bit;
}

template <class Ask>
inline uint64_t recover_hash_base_low_bits(uint64_t mask, int lowest_bit,
                                           Ask ask) {
    int highest_bit = highest_set_bit(mask);
    if (lowest_bit < 0) {
        lowest_bit = 0;
    }
    if (lowest_bit >= highest_bit) {
        return 0;
    }

    uint64_t recovered = 0;
    for (int bit_index = highest_bit - 1; bit_index >= lowest_bit;
         --bit_index) {
        int next_mask_bit = bit_index + 1;
        while (((mask >> next_mask_bit) & 1ULL) == 0) {
            ++next_mask_bit;
        }

        uint64_t query_offset = 0;
        if (bit_index + 1 < highest_bit) {
            int width = highest_bit - (bit_index + 1);
            uint64_t modulus = 1ULL << width;
            uint64_t current =
                (recovered >> (bit_index + 1)) & (modulus - 1);
            uint64_t desired = current;

            for (int bit = bit_index + 1;
                 bit < next_mask_bit && bit < highest_bit; ++bit) {
                desired |= 1ULL << (bit - (bit_index + 1));
            }
            if (next_mask_bit < highest_bit) {
                desired &= ~(1ULL << (next_mask_bit - (bit_index + 1)));
            }

            uint64_t delta = (desired + modulus - current) & (modulus - 1);
            query_offset = delta << (bit_index + 1);
        }

        int response =
            ask(query_offset, query_offset + (1ULL << bit_index)) & 1;
        int recovered_bit =
            response ^ static_cast<int>((mask >> bit_index) & 1ULL);
        if (recovered_bit != 0) {
            recovered |= 1ULL << bit_index;
        }
    }
    return recovered & low_bits_mask(highest_bit);
}

struct ProbeConfig {
    int device_id = 0;
    int target_smid = 0;
    int threshold_samples = 512;
    int validation_queries = 32;
    bool verbose = false;
};

struct RecoveryResult {
    uint64_t hash_base = 0;
    double threshold_cycles = 0.0;
    std::vector<int> sm_partition;
};

__global__ void latency_kernel(const uint8_t* address, uint64_t* times,
                               int* seen, int target_smid, uint64_t *_sum) {
    if (threadIdx.x != 0) {
        return;
    }

    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid != target_smid || atomicCAS(seen, 0, 1) != 0) {
        return;
    }

    unsigned int value __attribute__((unused));
#pragma unroll 1
    for (int trial = 0; trial < kLatencyTrials; ++trial) {
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(address));
        uint64_t begin = clock64();
        asm volatile("ld.global.volatile.u8 %0, [%1];"
                     : "=r"(value)
                     : "l"(address));
        *_sum += value;
        uint64_t end = clock64();
        times[trial] = end - begin;
    }
}

class LatencyProbe {
  public:
    LatencyProbe(int device_id, int target_smid, int sm_count)
        : target_smid_(target_smid), sm_count_(sm_count) {
        HFUSE_CUDA_CHECK(cudaSetDevice(device_id));
        if (target_smid < 0 || target_smid >= sm_count) {
            fail("target SMID is outside the device SM range");
        }
        HFUSE_CUDA_CHECK(
            cudaMalloc(&d_times_, kLatencyTrials * sizeof(uint64_t)));
        HFUSE_CUDA_CHECK(
            cudaMalloc(&d_sum_, sizeof(uint64_t)));
        try {
            HFUSE_CUDA_CHECK(cudaMalloc(&d_seen_, sizeof(int)));
            cudaDeviceProp properties{};
            HFUSE_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));
        } catch (...) {
            cudaFree(d_times_);
            cudaFree(d_seen_);
            cudaFree(d_sum_);
            d_times_ = nullptr;
            d_seen_ = nullptr;
            d_sum_ = nullptr;
            throw;
        }
    }

    ~LatencyProbe() {
        if (d_times_ != nullptr) {
            cudaFree(d_times_);
        }
        if (d_seen_ != nullptr) {
            cudaFree(d_seen_);
        }
        if (d_sum_ != nullptr) {
            cudaFree(d_sum_);
        }
    }

    double measure(const void* address, int smid) {
        if (smid < 0 || smid >= sm_count_) {
            fail("requested SMID is outside the device SM range");
        }

        uint64_t host_times[kLatencyTrials] = {};
        int blocks = std::max(sm_count_ * 4, smid + 1);
        for (int attempt = 0; attempt < 8; ++attempt) {
            HFUSE_CUDA_CHECK(cudaMemset(
                d_times_, 0, kLatencyTrials * sizeof(uint64_t)));
            HFUSE_CUDA_CHECK(cudaMemset(d_seen_, 0, sizeof(int)));
            HFUSE_CUDA_CHECK(cudaMemset(d_sum_, 0, sizeof(uint64_t)));
            latency_kernel<<<blocks, 1>>>(
                static_cast<const uint8_t*>(address), d_times_, d_seen_, smid, d_sum_);
            HFUSE_CUDA_CHECK(cudaGetLastError());
            HFUSE_CUDA_CHECK(cudaDeviceSynchronize());

            int seen = 0;
            HFUSE_CUDA_CHECK(cudaMemcpy(
                &seen, d_seen_, sizeof(int), cudaMemcpyDeviceToHost));
            if (seen == 0) {
                continue;
            }

            HFUSE_CUDA_CHECK(cudaMemcpy(host_times, d_times_,
                                        sizeof(host_times),
                                        cudaMemcpyDeviceToHost));
            uint64_t best = std::numeric_limits<uint64_t>::max();
            for (uint64_t value : host_times) {
                if (value != 0) {
                    best = std::min(best, value);
                }
            }
            if (best != std::numeric_limits<uint64_t>::max()) {
                return static_cast<double>(best);
            }
        }
        fail("could not schedule latency probe on SM " +
             std::to_string(smid));
    }

    double measure(const void* address) {
        return measure(address, target_smid_);
    }

  private:
    int target_smid_ = 0;
    int sm_count_ = 0;
    uint64_t* d_times_ = nullptr;
    int* d_seen_ = nullptr;
    uint64_t* d_sum_ = nullptr;
};

inline double estimate_threshold(LatencyProbe& probe, const uint8_t* base,
                                 size_t bytes, int sample_count,
                                 bool verbose) {
    if (sample_count < 2) {
        fail("threshold_samples must be at least 2");
    }
    uint64_t line_count = bytes / kProbeLineSize;
    if (line_count < 2) {
        fail("allocation is too small for latency sampling");
    }
    sample_count =
        static_cast<int>(std::min<uint64_t>(sample_count, line_count));

    std::vector<double> latencies;
    latencies.reserve(sample_count);
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int sample = 0; sample < sample_count; ++sample) {
        state = state * 2862933555777941757ULL + 3037000493ULL;
        uint64_t line = state % line_count;
        latencies.push_back(probe.measure(base + line * kProbeLineSize));
    }

    std::sort(latencies.begin(), latencies.end());
    double p25 = latencies[latencies.size() / 4];
    double p75 = latencies[(latencies.size() * 3) / 4];
    double threshold = 0.5 * (p25 + p75);
    if (verbose) {
        std::printf("[hfuse] latency threshold %.2f cycles "
                    "(min %.2f, p25 %.2f, p75 %.2f, max %.2f)\n",
                    threshold, latencies.front(), p25, p75,
                    latencies.back());
    }
    return threshold;
}

inline int classify(LatencyProbe& probe, const uint8_t* base, size_t bytes,
                    uint64_t offset, double threshold, int smid) {
    if (offset >= bytes) {
        fail("latency query is outside the allocation");
    }
    return probe.measure(base + offset, smid) > threshold ? 1 : 0;
}

inline int query_xor(LatencyProbe& probe, const uint8_t* base, size_t bytes,
                     uint64_t first, uint64_t second, double threshold,
                     int smid) {
    return classify(probe, base, bytes, first, threshold, smid) ^
           classify(probe, base, bytes, second, threshold, smid);
}

inline void validate_hash_base(LatencyProbe& probe, const uint8_t* base,
                               size_t bytes, uint64_t mask,
                               uint64_t hash_base, double threshold,
                               int target_smid, int query_count) {
    if (query_count <= 0) {
        fail("validation_queries must be positive");
    }
    size_t window_bytes = required_allocation_bytes(mask);
    if (window_bytes < 2 * kPartitionPageSize) {
        fail("hash-recovery window is too small for page validation");
    }
    uint64_t total_page_count = bytes >> kPartitionPageBits;
    if (total_page_count < 2) {
        fail("allocation is too small for page validation");
    }

    uint64_t state = 0x123456789abcdef0ULL;
    size_t window_count = (bytes + window_bytes - 1) / window_bytes;
    for (size_t window = 0; window < window_count; ++window) {
        uint64_t window_begin = window * window_bytes;
        uint64_t window_end =
            std::min<uint64_t>(bytes, window_begin + window_bytes);
        uint64_t page_count =
            (window_end - window_begin) >> kPartitionPageBits;
        if (page_count < 2) {
            fail("validation window has fewer than two complete pages");
        }

        for (int query = 0; query < query_count; ++query) {
            state = state * 2862933555777941757ULL + 3037000493ULL;
            uint64_t first =
                window_begin +
                ((state % page_count) << kPartitionPageBits);
            state = state * 2862933555777941757ULL + 3037000493ULL;
            uint64_t second =
                window_begin +
                ((state % page_count) << kPartitionPageBits);
            int measured = query_xor(probe, base, bytes, first, second,
                                     threshold, target_smid);
            int expected = partition_hash(mask, hash_base + first) ^
                           partition_hash(mask, hash_base + second);
            if (measured != expected) {
                fail("random hash validation failed in recovery window " +
                     std::to_string(window) +
                     "; the cudaMalloc range may not be hash-consistent or "
                     "the latency threshold is unstable");
            }
        }

        uint64_t page_skip =
            (((hash_base + window_begin) >> kPartitionPageBits) & 1ULL)
                ? 1
                : 0;
        uint64_t pair_count = (page_count - page_skip) / 2;
        if (pair_count == 0) {
            fail("validation window has no complete aligned page pair");
        }
        for (int query = 0; query < query_count; ++query) {
            state = state * 2862933555777941757ULL + 3037000493ULL;
            uint64_t first =
                window_begin +
                ((page_skip + 2 * (state % pair_count))
                 << kPartitionPageBits);
            uint64_t second = first + kPartitionPageSize;
            int measured = query_xor(probe, base, bytes, first, second,
                                     threshold, target_smid);
            int expected = partition_hash(mask, hash_base + first) ^
                           partition_hash(mask, hash_base + second);
            if (measured != 1 || expected != 1) {
                fail("consecutive-page validation failed in recovery window " +
                     std::to_string(window));
            }
        }

        uint64_t first = window_begin;
        uint64_t last =
            window_begin + ((page_count - 1) << kPartitionPageBits);
        int measured = query_xor(probe, base, bytes, first, last,
                                 threshold, target_smid);
        int expected = partition_hash(mask, hash_base + first) ^
                       partition_hash(mask, hash_base + last);
        if (measured != expected) {
            fail("window endpoint validation failed in recovery window " +
                 std::to_string(window));
        }

        if (window_begin != 0) {
            first = window_begin - kPartitionPageSize;
            uint64_t second = window_begin;
            measured = query_xor(probe, base, bytes, first, second,
                                 threshold, target_smid);
            expected = partition_hash(mask, hash_base + first) ^
                       partition_hash(mask, hash_base + second);
            if (measured != expected) {
                fail("hash validation failed across recovery-window boundary " +
                     std::to_string(window));
            }
        }
    }
}

inline RecoveryResult recover_and_probe(uint8_t* base, size_t bytes,
                                        uint64_t mask, int sm_count,
                                        const ProbeConfig& config) {
    if (base == nullptr) {
        fail("allocation base is null");
    }
    if ((mask & (kPartitionPageSize - 1)) != 0 ||
        (mask & kPartitionPageSize) == 0) {
        fail("partition mask is incompatible with 4 KiB page pairing");
    }
    if ((reinterpret_cast<uintptr_t>(base) &
         (kPartitionPageSize - 1)) != 0) {
        fail("cudaMalloc base is not 4 KiB aligned");
    }
    size_t recovery_bytes = required_allocation_bytes(mask);
    if (bytes < recovery_bytes) {
        fail("cudaMalloc allocation is too small for hash recovery");
    }

    LatencyProbe probe(config.device_id, config.target_smid, sm_count);
    double threshold = estimate_threshold(
        probe, base, recovery_bytes, config.threshold_samples,
        config.verbose);
    auto ask = [&](uint64_t first, uint64_t second) {
        return query_xor(probe, base, recovery_bytes, first, second, threshold,
                         config.target_smid);
    };
    int highest_bit = highest_set_bit(mask);
    uint64_t hash_base =
        recover_hash_base_low_bits(mask, kPartitionPageBits, ask);
    hash_base &= low_bits_mask(highest_bit);
    hash_base &= ~(kPartitionPageSize - 1);
    validate_hash_base(probe, base, bytes, mask, hash_base, threshold,
                       config.target_smid, config.validation_queries);

    uint64_t page_skip =
        ((hash_base >> kPartitionPageBits) & 1ULL)
            ? kPartitionPageSize
            : 0;
    uint64_t first = page_skip;
    uint64_t second = page_skip + kPartitionPageSize;
    int first_label = partition_hash(mask, hash_base + first);
    int second_label = partition_hash(mask, hash_base + second);
    if (first_label == second_label || second >= bytes) {
        fail("invalid page pair for SM affinity probing");
    }

    std::vector<int> affinity(sm_count, -1);
    int count_p0 = 0;
    int count_p1 = 0;
    for (int smid = 0; smid < sm_count; ++smid) {
        double first_time = probe.measure(base + first, smid);
        double second_time = probe.measure(base + second, smid);
        affinity[smid] =
            first_time < second_time ? first_label : second_label;
        count_p0 += affinity[smid] == 0;
        count_p1 += affinity[smid] == 1;
        if (config.verbose) {
            std::printf("[hfuse] SM %d: label%d %.2f cycles, "
                        "label%d %.2f cycles -> P%d\n",
                        smid, first_label, first_time, second_label,
                        second_time, affinity[smid]);
        }
    }
    if (count_p0 == 0 || count_p1 == 0) {
        fail("SM affinity probing produced an empty partition group");
    }

    return RecoveryResult{hash_base, threshold, std::move(affinity)};
}

}  // namespace hfuse
