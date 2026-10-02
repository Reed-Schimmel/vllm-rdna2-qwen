# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA/HIP memory helpers: pinning and batch DMA transfers."""

import ctypes
import mmap
import os
from typing import Any, NamedTuple

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# CUmemcpySrcAccessOrder values (CUDA driver API). STREAM(1): source read in
# stream order, safe when the source may still be written. ANY(3): source may
# be read early, only safe for a stable source (e.g. pinned host memory).
CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister.

    This bypasses PyTorch's CUDACachingHostAllocator which rounds
    every ``pin_memory=True`` allocation up to the next power of 2
    (e.g. 100 GB becomes 128 GB).
    """
    # gfx1030 fork: on ROCm each registration is one KFD userptr BO whose
    # pages stay movable. When the kernel migrates any page of it (compaction),
    # KFD evicts every GPU queue of the process and re-faults + remaps the
    # WHOLE BO before restoring them: 12 GB -> 12-20 s with no GPU progress
    # on that rank (measured 2026-10-01, /sys/class/kfd/kfd/proc/<pid>/
    # stats_*/evicted_ms). Registering in block-aligned chunks bounds each
    # stall to one chunk. Copies are per block, so none crosses a chunk.
    # VLLM_RDNA_OFFLOAD_PIN_CHUNK_MB=0 registers the tensor in one piece.
    chunk_mb = int(os.getenv("VLLM_RDNA_OFFLOAD_PIN_CHUNK_MB", "128"))
    block_bytes = tensor.stride(0) * tensor.element_size() if tensor.dim() else 0
    if current_platform.is_rocm() and chunk_mb > 0 and block_bytes > 0:
        step = max(1, (chunk_mb << 20) // block_bytes) * block_bytes
    else:
        step = tensor.nbytes
    base, n = tensor.data_ptr(), 0
    for off in range(0, tensor.nbytes, step):
        size = min(step, tensor.nbytes - off)
        err = torch.cuda.cudart().cudaHostRegister(base + off, size, 0)
        if err.value != 0:
            raise RuntimeError(f"cudaHostRegister failed at offset {off}: {err}")
        n += 1
    if n > 1:
        logger.info("Pinned %.2f GB CPU offload tensor as %d registrations of %.0f MB",
                    tensor.nbytes / 2**30, n, step / 2**20)


_thp_maps: list[mmap.mmap] = []


def zeros_thp(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Zeroed CPU tensor on transparent huge pages (gfx1030 fork).

    Private anonymous mapping, MADV_HUGEPAGE before first touch, so it is
    backed by 2 MB pages even with THP in "madvise" mode. Compaction skips
    compound pages of at least the order it is building (proactive compaction
    builds 2 MB), so these pages are not migrated under it -- which on ROCm
    would invalidate the registered range and evict all GPU queues (see
    pin_tensor). Falls back silently to 4 KB pages if THP is unavailable.
    """
    nbytes = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        m.madvise(mmap.MADV_HUGEPAGE)
    except (AttributeError, OSError) as e:
        logger.warning("MADV_HUGEPAGE failed (%s); CPU offload tier on 4 KB pages", e)
    _thp_maps.append(m)  # lives until process exit
    t = torch.frombuffer(m, dtype=torch.uint8).view(dtype).view(shape)

    def anon_huge_kb() -> int:
        with open("/proc/self/smaps_rollup") as f:
            return next((int(ln.split()[1]) for ln in f if ln.startswith("AnonHugePages")), 0)

    before = anon_huge_kb()
    t.zero_()  # fault in now (huge pages), not during serving
    huge = (anon_huge_kb() - before) << 10
    logger.info("CPU offload tensor %.2f GB: %.0f %% on 2 MB pages", nbytes / 2**30,
                100.0 * huge / max(1, nbytes))
    # Huge pages alone leave the 4 KB remainder movable, and direct compaction
    # (any high-order allocation on the host) still migrates it. mlock plus
    # vm.compact_unevictable_allowed=0 makes compaction skip the whole tier.
    # Needs RLIMIT_MEMLOCK >= the tier size; otherwise this only logs.
    if os.getenv("VLLM_RDNA_OFFLOAD_MLOCK", "1") == "1":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.mlock(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(nbytes)) != 0:
            logger.warning(
                "mlock of the CPU offload tier failed (%s; RLIMIT_MEMLOCK too low?): "
                "its 4 KB pages stay movable, and each migration evicts this "
                "rank's GPU queues", os.strerror(ctypes.get_errno()))
        else:
            try:
                with open("/proc/sys/vm/compact_unevictable_allowed") as f:
                    allowed = f.read().strip() == "1"
            except OSError:
                allowed = False
            logger.info("CPU offload tier mlocked%s", " (but vm.compact_unevictable_allowed=1: "
                        "compaction may still migrate it)" if allowed else "")
    return t


def lock_worker_memory() -> None:
    """mlockall(CURRENT|FUTURE|ONFAULT) for this GPU worker (gfx1030 fork).

    Besides the tier, ROCr keeps ~2.5 GB of its own host allocations per worker
    (hipHostMalloc pools, staging: ~1250 /dev/zero shared mappings), each a
    movable KFD userptr; a compaction burst over them still evicted the
    queues for 0.3-1.5 s. Locking every page the worker has or will touch
    (ONFAULT: nothing untouched is populated) makes them unmovable too, given
    vm.compact_unevictable_allowed=0. Gated with VLLM_RDNA_OFFLOAD_MLOCK.
    """
    if os.getenv("VLLM_RDNA_OFFLOAD_MLOCK", "1") != "1":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    MCL_CURRENT, MCL_FUTURE, MCL_ONFAULT = 1, 2, 4
    if libc.mlockall(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT) != 0:
        logger.warning("mlockall failed (%s; RLIMIT_MEMLOCK too low?): ROCr host "
                       "allocations stay movable", os.strerror(ctypes.get_errno()))
    else:
        logger.info("GPU worker memory mlocked (on fault)")


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


_BATCH_MEMCPY_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,  # CUresult / hipError_t
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
)

# Resolved lazily on first use: (entry point, numAttrs to pass).
_batch_memcpy: tuple[Any, int] | None = None

# Max copy descriptors per batch call, resolved lazily on first use.
_max_batch_descriptors: int | None = None

# ROCm hipMemcpyBatchAsync faults for count > 8192 (MI350X/gfx950), so we chunk
# at that ceiling. Override via VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS.
_ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS = 8192


def _resolve_max_batch_descriptors() -> int:
    """Max copy descriptors to pass to one batch-memcpy call (0 = unlimited).

    ROCm's ``hipMemcpyBatchAsync`` faults above 8192 descriptors per call, so
    on ROCm we cap and chunk larger transfers. CUDA's ``cuMemcpyBatchAsync``
    handles arbitrary counts and is left uncapped. Set
    ``VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS`` (>0) to override on any platform.
    """
    global _max_batch_descriptors
    if _max_batch_descriptors is None:
        override = envs.VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS
        if override > 0:
            _max_batch_descriptors = override
        else:
            _max_batch_descriptors = (
                _ROCM_DEFAULT_MAX_BATCH_DESCRIPTORS if current_platform.is_rocm() else 0
            )
    return _max_batch_descriptors


def _resolve_batch_memcpy() -> tuple[Any, int]:
    """Resolve the batch-memcpy entry point and its ``numAttrs`` (one-time).

    CUDA uses ``cuMemcpyBatchAsync``; ROCm uses ``hipMemcpyBatchAsync``.
    Raises ``RuntimeError`` if the symbol is unavailable (old CUDA driver,
    ROCm < 7.1, unusual install).
    """
    if current_platform.is_rocm():
        try:
            lib = _load_hip_runtime()
            fn = lib.hipMemcpyBatchAsync
        except (OSError, AttributeError) as e:
            raise RuntimeError(
                "hipMemcpyBatchAsync is unavailable in this ROCm install; "
                "SimpleCPUOffloadConnector requires ROCm 7.1+."
            ) from e
        fn.restype = ctypes.c_uint
        fn.argtypes = [
            ctypes.c_void_p,  # dsts
            ctypes.c_void_p,  # srcs
            ctypes.c_void_p,  # sizes
            ctypes.c_size_t,  # count
            ctypes.c_void_p,  # attrs
            ctypes.c_void_p,  # attrIdxs
            ctypes.c_size_t,  # numAttrs
            ctypes.c_void_p,  # failIdx
            ctypes.c_void_p,  # stream
        ]
        return fn, _rocm_num_attrs(lib)

    from cuda.bindings import driver as drv

    err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGetProcAddress(cuMemcpyBatchAsync) failed: {err}")
    return _BATCH_MEMCPY_FUNC_TYPE(ptr), 1


def _load_hip_runtime() -> ctypes.CDLL:
    """Load ``libamdhip64``, tolerating installs without the devel symlink.

    The unversioned ``libamdhip64.so`` only ships with the ROCm devel package;
    runtime-only and wheel-packaged ROCm installs provide just the versioned
    soname. ``dlopen`` returns the already-mapped library when asked for a
    soname the process has loaded — torch loads HIP at import — so the
    versioned names resolve even when they are not on the loader search path.
    """
    errors = []
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            errors.append(f"{name}: {e}")
    raise OSError("could not load the HIP runtime: " + "; ".join(errors))


def _rocm_num_attrs(lib: ctypes.CDLL) -> int:
    """``numAttrs`` for ``hipMemcpyBatchAsync`` on the running HIP runtime."""
    ver = ctypes.c_int(0)
    try:
        if lib.hipRuntimeGetVersion(ctypes.byref(ver)) != 0:
            ver.value = 0
    except (OSError, AttributeError):
        ver.value = 0
    return _num_attrs_for_hip_version(ver.value)


def _num_attrs_for_hip_version(version: int) -> int:
    """``numAttrs`` for ``hipMemcpyBatchAsync`` given a HIP runtime version int.

    ROCm 7.2.1-7.2.3 reject ``numAttrs > 0`` (ROCm/clr @ rocm-7.2.1
    hipamd/src/hip_memory.cpp:2819-2822); 7.13+ accept it. ``version`` 0
    (unknown) yields the conservative 0.
    """
    # HIP encodes version as major*10_000_000 + minor*100_000 + patch.
    major, minor = version // 10_000_000, (version // 100_000) % 100
    return 1 if (major, minor) >= (7, 13) else 0


class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray  # [num_layers] uint64 — data_ptr per layer
    dst_bases: np.ndarray  # [num_layers] uint64
    bpb: np.ndarray  # [num_layers] uint64 — bytes per block
    num_layers: int
    # One attributes entry carrying srcAccessOrder. Ignored when num_attrs is
    # 0, which is what ROCm runtimes older than 7.13 require (see
    # _num_attrs_for_hip_version).
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t
    num_attrs: int
    # NOTE: cuMemcpyBatchAsync_v2() removed fail_idx field, but we use
    # cuMemcpyBatchAsync() with fail_idx for backward compatibility
    fail_idx: ctypes.c_size_t
    stream_handle: int  # raw cudaStream_t / CUstream


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
    src_access_order: int = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
) -> BatchMemcpyParams:
    global _batch_memcpy
    if _batch_memcpy is None:
        _batch_memcpy = _resolve_batch_memcpy()
    _, num_attrs = _batch_memcpy

    assert list(src_caches.keys()) == list(dst_caches.keys())
    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())

    src_bases, dst_bases, bpb = [], [], []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        assert s_bpb == d.stride(0) * d.element_size()
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    attrs = _CUmemcpyAttributes(srcAccessOrder=src_access_order)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
        num_attrs=num_attrs,
        fail_idx=ctypes.c_size_t(0),
        stream_handle=stream.cuda_stream,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Copy blocks via cuMemcpyBatchAsync / hipMemcpyBatchAsync."""
    n = len(src_block_ids)
    if n == 0:
        return

    assert _batch_memcpy is not None, "build_params() must run before copy_blocks()"
    fn, _ = _batch_memcpy

    src_ids = np.array(src_block_ids, dtype=np.uint64)
    dst_ids = np.array(dst_block_ids, dtype=np.uint64)

    src_all = (
        params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
    ).ravel()
    dst_all = (
        params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
    ).ravel()
    sz_all = np.repeat(params.bpb, n)
    total = n * params.num_layers

    # Chunk on ROCm: hipMemcpyBatchAsync faults above 8192 descriptors/call.
    # CUDA is uncapped (max_desc == 0) and issues a single call.
    max_desc = _resolve_max_batch_descriptors()
    step = total if max_desc <= 0 else max_desc
    for off in range(0, total, step):
        cnt = min(step, total - off)
        err = fn(
            dst_all[off : off + cnt].ctypes.data,
            src_all[off : off + cnt].ctypes.data,
            sz_all[off : off + cnt].ctypes.data,
            cnt,
            ctypes.addressof(params.attrs),
            ctypes.byref(params.attrs_idx),
            params.num_attrs,
            ctypes.byref(params.fail_idx),
            params.stream_handle,
        )
        if err != 0:
            raise RuntimeError(
                f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
            )
