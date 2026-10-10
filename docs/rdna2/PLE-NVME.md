# Experimental INT4 PLE NVMe cache

This opt-in path keeps the existing INT4 quantization and GPU transport, but
serves embedding rows through the CPU NVMe cache recovered from
`tcclaviger/vllm:29.07.1`. The default RAM-backed sidecar path is unchanged.

**Status: implementation draft. Native compilation, correctness tests, model
validation and performance measurements are pending. Do not treat this as a
validated production configuration.** Coordinate resource-intensive work with
any other benchmark or serving session before running the commands below.

## What is shared with RDNA4

The imported `ple_nvme_ext.NvmeTable` extension is pure Linux C++: O_DIRECT
reads, raw io_uring syscalls, a pread-thread-pool fallback, miss deduplication,
and clock replacement. It does not import torch or require ROCm, HIP, liburing
or a GPU compiler. The direct-ring GPU transport is not used.

Reference provenance:

- Image manifest:
  `sha256:a7b08ce3268ed645bd8154e5df3e86615a61d269c416d8511876674a272b3b53`.
- Source: `/app/src/plehip/csrc/ple_nvme.cpp` in that image; SPDX Apache-2.0.
- Source SHA256:
  `7f42d0fc4274e8d972a94c39b038c6b2d3217a5b081536b50d553faec0d5b2e6`.
- The source reference is retained separately at
  `~/vLLM/RDNA4-RDNA2-merge/rootfs/app/src/plehip/csrc/ple_nvme.cpp`.

We reuse that extension unchanged. The RDNA2 adapter accepts only the existing
`group16_int4_fp16scale_lownibblefirst` sidecar. Each logical 160-element row is
stored as 80 packed bytes followed by ten FP16 scales (20 bytes). The exporter
interleaves those bytes without decoding or requantization. A derived table
therefore takes about 32 GB of **additional disk space**, not 32 GB of RAM.

## Build the optional CPU dependency

Use the isolated worktree's own `.venv`, never the other session's environment.
Complete the normal editable-install/lint setup described by `AGENTS.md` first,
with hook installation isolated from other worktrees. Then:

```bash
uv pip install --python .venv/bin/python pybind11
PLE_NVME_SOURCE="$HOME/vLLM/RDNA4-RDNA2-merge/rootfs/app/src/plehip/csrc/ple_nvme.cpp" \
  bash tools/rdna2/build-ple-nvme.sh
```

Only this optional CPU extension is compiled. It is installed in this worktree's
`.venv`, not in the source package or a shared serving environment. The helper
checks the inspected source hash; do not override it without reviewing changes.
There is no import-time JIT or automatic build during server startup.

## Prepare a table explicitly

**This step reads the whole sidecar and writes about 32 GB.** It also advises
source/output file pages out of the filesystem cache. Run it only in an agreed
testing window, never alongside benchmarks that use those same source files.
Use a real NVMe directory: this host's `/tmp` is tmpfs.

```bash
.venv/bin/python tools/rdna2/prepare-ple-nvme.py \
  --sidecar /path/to/ples_int4 \
  --output /path/on/nvme/ple-int4-cache
```

The exporter streams small chunks and writes `ple_int4.bin` and
`ple_int4.json`. It retains a checksum of the interleaved row data and pads the
file tail for direct I/O. It does not modify the original sidecar.

Matching prepared tables are reused. Reuse validates metadata, source file
identity (path, size, inode, timestamps), and derived-file identity/size; it does
not rescan all 32 GB on every startup. Changed or incomplete tables are rejected,
not silently overwritten. Use a new dedicated output directory to rebuild.
Keep source and derived files immutable while serving. One process owns each
derived-table directory through an exclusive file lock.

## Serve

```bash
MODEL=/path/to/awq-backbone \
PLE_INT4=/path/to/ples_int4 \
PLE_NVME_DIR=/path/on/nvme/ple-int4-cache \
PLE_NVME_CACHE_GB=8 \
PLE_OFFLOAD_ANON=0 \
  bash tools/rdna2/serve-qwen38-flash-next.sh
```

| Serve knob | Default | Meaning |
| --- | --- | --- |
| `PLE_NVME_DIR` | unset | Enable NVMe caching using an explicitly prepared table |
| `PLE_NVME_CACHE_GB` | 8 | GiB budget for row arena, index and estimated I/O staging |
| `PLE_NVME_QUEUE_DEPTH` | 512 | Outstanding miss-read budget, from 1 to 4096 |

These become `VLLM_PLE_NVME_DIR`, `VLLM_PLE_NVME_CACHE_GB` and
`VLLM_PLE_NVME_QUEUE_DEPTH` in the CPU worker. This mode requires the INT4
sidecar and cannot be combined with `PLE_DISK_DIR` or `PLE_OFFLOAD_ANON=1`.
Whole-table prefaulting is skipped regardless of `PLE_OFFLOAD_PREFAULT`.
The cache starts empty and learns from traffic; there is no 8 GiB warm-start read.

Unlike the RDNA4 flag, the budget includes the large row-ID index, not just row
payload. It is not a ceiling on total process RSS: batch buffers, io_uring
metadata, allocator overhead, thread stacks and the rest of vLLM are additional.
Direct-I/O alignment greater than 4096 is currently unsupported. The native
extension attempts mlock; check startup warnings and raise the memlock limit
before relying on cache residency under memory pressure.

Every 500 worker requests, cache hits, misses and distinct disk reads are logged
alongside the existing PLE lookup timing. NVMe misses cost latency, especially on
large prefills. No performance equivalence to full RAM is claimed yet.

To revert, unset `PLE_NVME_DIR` and use the original serve settings. Prepared
tables are retained on shutdown and are not required by the RAM-backed path.

## Pending validation

The existing `tests/v1/worker/test_ple_offload_worker.py` suite now includes tiny
export/decode/reuse/budget/locking tests and an optional native cache test. The
native test skips when `ple_nvme_ext` is not installed; that skip is not evidence
that native I/O works. Run the suite in the agreed testing window:

```bash
.venv/bin/python -m pytest tests/v1/worker/test_ple_offload_worker.py -v
```

Then compare RAM-backed and NVMe-backed operation using the existing PLE
consistency and serving validation tools. Measure cold/warm cache behavior,
long prefills, concurrency, total RAM and storage reads. Include error-path and
fallback testing of the imported native implementation before production use.

This change alone does not make the forks checkpoint-compatible or enable
disaggregated prefill. QSA and hybrid-state transfer remain separate work.
