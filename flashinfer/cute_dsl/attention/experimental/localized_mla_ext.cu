/*
 * Experimental B200/B300 localized allocator for partition-aware MLA.
 *
 * The RM definitions below are the minimal subset derived from NVIDIA's
 * MIT-licensed open-gpu-kernel-modules release 595.45.04.  This experiment is
 * deliberately fail-closed on other GPUs and driver ABIs.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace rmabi {

using Handle = std::uint32_t;
using U8 = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using S32 = std::int32_t;
using U64 = std::uint64_t;
using P64 = std::uint64_t;
using Status = std::uint32_t;

constexpr Status kOk = 0;
constexpr unsigned int kIoctlMagic = 'F';
constexpr unsigned int kEscRmFree = 0x29;
constexpr unsigned int kEscRmControl = 0x2A;
constexpr unsigned int kEscRmAlloc = 0x2B;
constexpr U32 kClassRootClient = 0x00000041;
constexpr U32 kClassDevice = 0x00000080;
constexpr U32 kClassSubdevice = 0x00002080;
constexpr U32 kClassMemoryLocalUser = 0x00000040;
constexpr U32 kCtrlGetGlobalSmOrder = 0x2080121b;
constexpr U32 kCtrlExportObjectToFd = 0x00003d05;
constexpr U32 kExportObjectTypeRm = 1;
constexpr U32 kMemoryTypeImage = 0;
constexpr U32 kAllocAlignmentForce = 0x00000100;
constexpr U32 kAttrPitchHugeContiguous = 0x11800000;
constexpr U32 kAttrPitchHugeNoncontiguous = 0x09800000;
constexpr U32 kAttr2Huge2MiB = 0x00100000;
constexpr U32 kAttr2LocalizedUgpu0 = 0x20000000;
constexpr U32 kAttr2LocalizedUgpu1 = 0x40000000;

struct FreeParams {
  Handle root;
  Handle parent;
  Handle object;
  Status status;
};

struct AllocParams {
  Handle root;
  Handle parent;
  Handle object;
  U32 class_id;
  alignas(8) P64 class_params;
  U32 class_params_size;
  Status status;
};

struct ControlParams {
  Handle client;
  Handle object;
  U32 command;
  U32 flags;
  alignas(8) P64 params;
  U32 params_size;
  Status status;
};

struct DeviceAllocParams {
  U32 device_id;
  Handle client_share;
  Handle target_client;
  Handle target_device;
  U32 flags;
  alignas(8) U64 va_space_size;
  alignas(8) U64 va_start_internal;
  alignas(8) U64 va_limit_internal;
  U32 va_mode;
};

struct SubdeviceAllocParams {
  U32 subdevice_id;
};

struct MemoryAllocParams {
  U32 owner;
  U32 type;
  U32 flags;
  U32 width;
  U32 height;
  S32 pitch;
  U32 attr;
  U32 attr2;
  U32 format;
  U32 compression_coverage;
  U32 zcull_coverage;
  alignas(8) U64 range_lo;
  alignas(8) U64 range_hi;
  alignas(8) U64 size;
  alignas(8) U64 alignment;
  alignas(8) U64 offset;
  alignas(8) U64 limit;
  alignas(8) P64 address;
  U32 ctag_offset;
  Handle va_space;
  U32 internal_flags;
  U32 tag;
  S32 numa_node;
};

struct alignas(8) GrRouteInfo {
  U32 flags;
  alignas(8) U64 route;
};

struct GlobalSmEntry {
  U16 gpc_id;
  U16 local_tpc_id;
  U16 local_sm_id;
  U16 global_tpc_id;
  U16 virtual_gpc_id;
  U16 migratable_tpc_id;
  U16 ugpu_id;
  U16 physical_cpc_id;
  U16 virtual_tpc_id;
};

struct GlobalSmOrderParams {
  GlobalSmEntry global_sm_id[512];
  U16 num_sm;
  U16 num_tpc;
  alignas(8) GrRouteInfo route_info;
};

struct ExportObject {
  U32 type;
  union {
    struct {
      Handle device;
      Handle parent;
      Handle object;
    } rm_object;
  } data;
};

struct ExportObjectToFdParams {
  ExportObject object;
  S32 fd;
  U32 flags;
};

static_assert(sizeof(FreeParams) == 16);
static_assert(sizeof(AllocParams) == 32);
static_assert(sizeof(ControlParams) == 32);
static_assert(sizeof(DeviceAllocParams) == 56);
static_assert(sizeof(SubdeviceAllocParams) == 4);
static_assert(sizeof(MemoryAllocParams) == 128);
static_assert(sizeof(GrRouteInfo) == 16);
static_assert(sizeof(GlobalSmEntry) == 18);
static_assert(sizeof(GlobalSmOrderParams) == 9240);
static_assert(sizeof(ExportObject) == 16);
static_assert(sizeof(ExportObjectToFdParams) == 24);

}  // namespace rmabi

namespace {

constexpr std::uint64_t kHugePageBytes = 2ULL << 20;
constexpr std::uint64_t kMaxLocalizedContiguousBytes = 32ULL << 20;

std::string cuda_driver_error(CUresult result, const char* operation) {
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::ostringstream stream;
  stream << operation << ": "
         << (name != nullptr ? name : "unknown CUDA error") << " ("
         << static_cast<int>(result) << ")";
  if (message != nullptr) stream << ": " << message;
  return stream.str();
}

void cuda_driver_check(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(cuda_driver_error(result, operation));
  }
}

void cuda_runtime_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

unsigned long rm_ioctl_request(unsigned int escape, std::size_t size) {
  return _IOC(_IOC_READ | _IOC_WRITE, rmabi::kIoctlMagic, escape, size);
}

void rm_ioctl_checked(int fd, unsigned int escape, void* params,
                      std::size_t size, rmabi::Status* status,
                      const char* operation) {
  errno = 0;
  int const rc = ioctl(fd, rm_ioctl_request(escape, size), params);
  if (rc != 0) {
    std::ostringstream stream;
    stream << operation << ": ioctl failed with errno " << errno << " ("
           << std::strerror(errno) << ")";
    throw std::runtime_error(stream.str());
  }
  if (*status != rmabi::kOk) {
    std::ostringstream stream;
    stream << operation << ": RM status 0x" << std::hex << *status;
    throw std::runtime_error(stream.str());
  }
}

std::uint64_t pointer_to_p64(void* pointer) {
  return static_cast<std::uint64_t>(
      reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (value > std::numeric_limits<std::uint64_t>::max() - alignment + 1) {
    throw std::overflow_error("localized allocation size overflow");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

std::string normalize_pci_bus_id(std::string bus_id) {
  std::transform(bus_id.begin(), bus_id.end(), bus_id.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::size_t const colon = bus_id.find(':');
  if (colon == 8) bus_id = bus_id.substr(4);
  return bus_id;
}

int rm_minor_for_cuda_device(CUdevice device) {
  std::array<char, 32> pci_bus_id{};
  cuda_driver_check(
      cuDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), device),
      "cuDeviceGetPCIBusId");
  std::string const information_path =
      "/proc/driver/nvidia/gpus/" +
      normalize_pci_bus_id(pci_bus_id.data()) + "/information";
  std::ifstream information(information_path);
  if (!information) {
    throw std::runtime_error("failed to open " + information_path);
  }
  std::string line;
  while (std::getline(information, line)) {
    std::size_t const label = line.find("Device Minor:");
    if (label == std::string::npos) continue;
    std::size_t const colon = line.find(':', label);
    return std::stoi(line.substr(colon + 1));
  }
  throw std::runtime_error("Device Minor is absent from " + information_path);
}

class CurrentContextGuard {
 public:
  explicit CurrentContextGuard(CUcontext context) {
    cuda_driver_check(cuCtxGetCurrent(&previous_), "cuCtxGetCurrent");
    if (previous_ != context) {
      cuda_driver_check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
      changed_ = true;
    }
  }
  ~CurrentContextGuard() {
    if (changed_) cuCtxSetCurrent(previous_);
  }

 private:
  CUcontext previous_ = nullptr;
  bool changed_ = false;
};

struct LocalizedMemory {
  rmabi::Handle rm_handle = 0;
  CUexternalMemory external_memory = nullptr;
  CUdeviceptr pointer = 0;
  std::uint64_t size = 0;
};

class LocalizedContext {
 public:
  explicit LocalizedContext(int cuda_device_id)
      : cuda_device_id_(cuda_device_id) {
    try {
      initialize();
    } catch (...) {
      close();
      throw;
    }
  }
  ~LocalizedContext() { close(); }
  LocalizedContext(LocalizedContext const&) = delete;
  LocalizedContext& operator=(LocalizedContext const&) = delete;

  std::vector<std::int32_t> const& sm_partition_map() const {
    return sm_partition_map_;
  }

  std::vector<std::array<std::int32_t, 6>> const& sm_topology() const {
    return sm_topology_;
  }

  std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>
  allocate(std::uint64_t bytes_p0, std::uint64_t bytes_p1) {
    if (closed_) throw std::runtime_error("localized context is closed");
    if (allocated_) throw std::runtime_error("localized pools already allocated");
    CurrentContextGuard guard(primary_context_);
    try {
      allocate_one(0, bytes_p0);
      allocate_one(1, bytes_p1);
      allocated_ = true;
    } catch (...) {
      release_memory(1);
      release_memory(0);
      throw;
    }
    return {static_cast<std::int64_t>(memory_[0].pointer),
            static_cast<std::int64_t>(memory_[0].size),
            static_cast<std::int64_t>(memory_[1].pointer),
            static_cast<std::int64_t>(memory_[1].size)};
  }

  void close() noexcept {
    if (closed_) return;
    if (primary_context_ != nullptr) {
      CUcontext previous = nullptr;
      cuCtxGetCurrent(&previous);
      cuCtxSetCurrent(primary_context_);
      release_memory(1);
      release_memory(0);
      cuCtxSetCurrent(previous);
    }
    if (client_ != 0) {
      rm_free_noexcept(0, client_);
      client_ = 0;
      device_ = 0;
      subdevice_ = 0;
    }
    if (device_fd_ >= 0) ::close(device_fd_);
    if (control_fd_ >= 0) ::close(control_fd_);
    device_fd_ = -1;
    control_fd_ = -1;
    if (primary_context_retained_) {
      cuDevicePrimaryCtxRelease(cuda_device_);
      primary_context_retained_ = false;
      primary_context_ = nullptr;
    }
    closed_ = true;
  }

 private:
  void initialize() {
    if (cuda_device_id_ < 0) {
      throw std::invalid_argument("device_id must be non-negative");
    }
    cuda_driver_check(cuInit(0), "cuInit");
    int device_count = 0;
    cuda_driver_check(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (cuda_device_id_ >= device_count) {
      throw std::invalid_argument("device_id is outside visible CUDA devices");
    }
    cuda_driver_check(cuDeviceGet(&cuda_device_, cuda_device_id_),
                      "cuDeviceGet");
    std::array<char, 128> device_name{};
    cuda_driver_check(
        cuDeviceGetName(device_name.data(), device_name.size(), cuda_device_),
        "cuDeviceGetName");
    std::string const name(device_name.data());
    if (name.find("B200") == std::string::npos &&
        name.find("B300") == std::string::npos) {
      throw std::runtime_error(
          "localized allocation supports only B200/B300; detected " + name);
    }
    cuda_driver_check(cuDevicePrimaryCtxRetain(&primary_context_, cuda_device_),
                      "cuDevicePrimaryCtxRetain");
    primary_context_retained_ = true;

    int const rm_minor = rm_minor_for_cuda_device(cuda_device_);
    control_fd_ = ::open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (control_fd_ < 0) {
      throw std::runtime_error("open /dev/nvidiactl failed: " +
                               std::string(std::strerror(errno)));
    }
    std::string const device_path = "/dev/nvidia" + std::to_string(rm_minor);
    device_fd_ = ::open(device_path.c_str(), O_RDWR | O_CLOEXEC);
    if (device_fd_ < 0) {
      throw std::runtime_error("open " + device_path + " failed: " +
                               std::string(std::strerror(errno)));
    }

    client_ = rm_alloc(0, rmabi::kClassRootClient, nullptr, 0,
                       "RM allocate root client");
    rmabi::DeviceAllocParams device_params{};
    device_params.device_id = static_cast<rmabi::U32>(rm_minor);
    device_ = rm_alloc(client_, rmabi::kClassDevice, &device_params,
                       sizeof(device_params), "RM allocate device");
    rmabi::SubdeviceAllocParams subdevice_params{};
    subdevice_ = rm_alloc(device_, rmabi::kClassSubdevice, &subdevice_params,
                          sizeof(subdevice_params), "RM allocate subdevice");

    rmabi::GlobalSmOrderParams sm_order{};
    rm_control(subdevice_, rmabi::kCtrlGetGlobalSmOrder, &sm_order,
               sizeof(sm_order), "RM query global SM order");
    int cuda_sm_count = 0;
    cuda_driver_check(
        cuDeviceGetAttribute(&cuda_sm_count,
                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                             cuda_device_),
        "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)");
    if (sm_order.num_sm == 0 || sm_order.num_sm > 512 ||
        sm_order.num_sm != cuda_sm_count) {
      throw std::runtime_error("RM/CUDA SM count mismatch");
    }
    std::array<int, 2> counts{};
    for (unsigned int sm = 0; sm < sm_order.num_sm; ++sm) {
      auto const& entry = sm_order.global_sm_id[sm];
      if (entry.ugpu_id > 1) {
        throw std::runtime_error("unsupported RM uGPU ID");
      }
      sm_partition_map_.push_back(static_cast<std::int32_t>(entry.ugpu_id));
      sm_topology_.push_back(
          {static_cast<std::int32_t>(entry.gpc_id),
           static_cast<std::int32_t>(entry.local_tpc_id),
           static_cast<std::int32_t>(entry.local_sm_id),
           static_cast<std::int32_t>(entry.global_tpc_id),
           static_cast<std::int32_t>(entry.physical_cpc_id),
           static_cast<std::int32_t>(entry.ugpu_id)});
      ++counts[entry.ugpu_id];
    }
    if (counts[0] == 0 || counts[1] == 0) {
      throw std::runtime_error("RM did not expose both B200/B300 uGPUs");
    }
  }

  rmabi::Handle rm_alloc(rmabi::Handle parent, rmabi::U32 class_id,
                         void* class_params,
                         rmabi::U32 class_params_size,
                         const char* operation) {
    rmabi::AllocParams params{};
    params.root = client_;
    params.parent = parent;
    params.class_id = class_id;
    params.class_params = pointer_to_p64(class_params);
    params.class_params_size = class_params_size;
    rm_ioctl_checked(control_fd_, rmabi::kEscRmAlloc, &params, sizeof(params),
                     &params.status, operation);
    return params.object;
  }

  void rm_control(rmabi::Handle object, rmabi::U32 command,
                  void* command_params, rmabi::U32 command_params_size,
                  const char* operation) {
    rmabi::ControlParams params{};
    params.client = client_;
    params.object = object;
    params.command = command;
    params.params = pointer_to_p64(command_params);
    params.params_size = command_params_size;
    rm_ioctl_checked(control_fd_, rmabi::kEscRmControl, &params,
                     sizeof(params), &params.status, operation);
  }

  void rm_free_noexcept(rmabi::Handle parent,
                        rmabi::Handle object) noexcept {
    if (control_fd_ < 0 || object == 0) return;
    rmabi::FreeParams params{};
    params.root = client_;
    params.parent = parent;
    params.object = object;
    ioctl(control_fd_, rm_ioctl_request(rmabi::kEscRmFree, sizeof(params)),
          &params);
  }

  void allocate_one(unsigned int ugpu, std::uint64_t requested_bytes) {
    if (requested_bytes == 0) return;
    std::uint64_t const allocation_bytes =
        align_up(requested_bytes, kHugePageBytes);
    rmabi::MemoryAllocParams memory_params{};
    memory_params.owner = 0x6e766c6fU;  // "nvlo"
    memory_params.type = rmabi::kMemoryTypeImage;
    memory_params.flags = rmabi::kAllocAlignmentForce;
    memory_params.attr = allocation_bytes > kMaxLocalizedContiguousBytes
                             ? rmabi::kAttrPitchHugeNoncontiguous
                             : rmabi::kAttrPitchHugeContiguous;
    memory_params.attr2 =
        rmabi::kAttr2Huge2MiB |
        (ugpu == 0 ? rmabi::kAttr2LocalizedUgpu0
                   : rmabi::kAttr2LocalizedUgpu1);
    memory_params.size = allocation_bytes;
    memory_params.alignment = kHugePageBytes;
    memory_[ugpu].rm_handle =
        rm_alloc(device_, rmabi::kClassMemoryLocalUser, &memory_params,
                 sizeof(memory_params), "RM allocate localized memory");
    memory_[ugpu].size = memory_params.size;

    int export_fd = ::open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (export_fd < 0) {
      throw std::runtime_error("open localized export FD failed: " +
                               std::string(std::strerror(errno)));
    }
    rmabi::ExportObjectToFdParams export_params{};
    export_params.object.type = rmabi::kExportObjectTypeRm;
    export_params.object.data.rm_object.device = device_;
    export_params.object.data.rm_object.parent = device_;
    export_params.object.data.rm_object.object = memory_[ugpu].rm_handle;
    export_params.fd = export_fd;
    try {
      rm_control(client_, rmabi::kCtrlExportObjectToFd, &export_params,
                 sizeof(export_params), "RM export localized memory FD");
      CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle_desc{};
      handle_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
      handle_desc.handle.fd = export_fd;
      handle_desc.size = memory_[ugpu].size;
      cuda_driver_check(
          cuImportExternalMemory(&memory_[ugpu].external_memory, &handle_desc),
          "cuImportExternalMemory(localized)");
      export_fd = -1;
      CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc{};
      buffer_desc.size = memory_[ugpu].size;
      cuda_driver_check(
          cuExternalMemoryGetMappedBuffer(&memory_[ugpu].pointer,
                                          memory_[ugpu].external_memory,
                                          &buffer_desc),
          "cuExternalMemoryGetMappedBuffer(localized)");
    } catch (...) {
      if (export_fd >= 0) ::close(export_fd);
      throw;
    }
  }

  void release_memory(unsigned int ugpu) noexcept {
    LocalizedMemory& memory = memory_[ugpu];
    if (memory.pointer != 0) cuMemFree(memory.pointer);
    if (memory.external_memory != nullptr)
      cuDestroyExternalMemory(memory.external_memory);
    if (memory.rm_handle != 0) rm_free_noexcept(device_, memory.rm_handle);
    memory = {};
  }

  int cuda_device_id_ = 0;
  CUdevice cuda_device_ = 0;
  CUcontext primary_context_ = nullptr;
  bool primary_context_retained_ = false;
  int control_fd_ = -1;
  int device_fd_ = -1;
  rmabi::Handle client_ = 0;
  rmabi::Handle device_ = 0;
  rmabi::Handle subdevice_ = 0;
  std::vector<std::int32_t> sm_partition_map_;
  std::vector<std::array<std::int32_t, 6>> sm_topology_;
  LocalizedMemory memory_[2];
  bool allocated_ = false;
  bool closed_ = false;
};

__global__ void __cluster_dims__(2, 1, 1)
cluster_probe_kernel(std::int32_t* block_to_smid) {
  if (threadIdx.x == 0) {
    std::uint32_t smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    block_to_smid[blockIdx.x] = static_cast<std::int32_t>(smid);
  }
}

std::vector<std::int32_t> probe_cluster_smids(int device_id, int sm_count) {
  if (sm_count <= 0 || (sm_count % 2) != 0) {
    throw std::invalid_argument("SM count must be positive and even");
  }
  cuda_runtime_check(cudaSetDevice(device_id), "cudaSetDevice");
  std::int32_t* output = nullptr;
  cuda_runtime_check(cudaMalloc(&output, sm_count * sizeof(std::int32_t)),
                     "cudaMalloc(cluster probe)");
  try {
    constexpr std::size_t dynamic_smem_bytes = 128 * 1024;
    cuda_runtime_check(
        cudaFuncSetAttribute(cluster_probe_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             dynamic_smem_bytes),
        "cudaFuncSetAttribute(cluster probe)");
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(sm_count, 1, 1);
    config.blockDim = dim3(32, 1, 1);
    config.dynamicSmemBytes = dynamic_smem_bytes;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cuda_runtime_check(cudaLaunchKernelEx(&config, cluster_probe_kernel, output),
                       "cudaLaunchKernelEx(cluster probe)");
    std::vector<std::int32_t> result(sm_count);
    cuda_runtime_check(cudaMemcpy(result.data(), output,
                                  sm_count * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(cluster probe)");
    cuda_runtime_check(cudaFree(output), "cudaFree(cluster probe)");
    return result;
  } catch (...) {
    cudaFree(output);
    throw;
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "experimental B200/B300 localized MLA runtime";
  py::class_<LocalizedContext>(module, "LocalizedContext")
      .def(py::init<int>(), py::arg("device_id"))
      .def("sm_partition_map", &LocalizedContext::sm_partition_map)
      .def("sm_topology", &LocalizedContext::sm_topology)
      .def("allocate", &LocalizedContext::allocate, py::arg("bytes_p0"),
           py::arg("bytes_p1"))
      .def("close", &LocalizedContext::close);
  module.def("probe_cluster_smids", &probe_cluster_smids,
             py::arg("device_id"), py::arg("sm_count"));
}
