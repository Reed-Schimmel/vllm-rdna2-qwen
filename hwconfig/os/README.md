# OS settings

Operating-system settings this fork depends on, as files laid out like their target paths. None of them is
needed for basic serving; each is needed for a specific feature, named below. Check a host with
[`../check-host.sh`](../check-host.sh).

| File | Install to | Needed for |
|---|---|---|
| [`etc/sysctl.d/90-vllm-offload.conf`](etc/sysctl.d/90-vllm-offload.conf) | `/etc/sysctl.d/` | CPU KV-cache tier (`--kv-offloading-size`) |
| [`etc/systemd/system/user@.service.d/memlock.conf`](etc/systemd/system/user@.service.d/memlock.conf) | `/etc/systemd/system/user@.service.d/` | CPU KV-cache tier, when vLLM runs under a systemd user unit |
| [`systemd-user/vllm.service.d/memlock.conf`](systemd-user/vllm.service.d/memlock.conf) | `~/.config/systemd/user/<your unit>.service.d/` | same |

The kernel command line is in [`../../PRODUCTION.md`](../../PRODUCTION.md#kernel-command-line); the amdgpu
patches and the per-boot GPU settings are in [`../kernel-patches/`](../kernel-patches/) and
[`../cardinit/`](../cardinit/).

## Memory settings for the CPU KV-cache tier

**Symptom when missing:** with `--kv-offloading-size`, occasional steps of 5–20 s on *all* tensor-parallel
ranks at once, `PLE lookup for launch N has taken >5 s` on the ranks waiting for the stalled one, and, if a
stall exceeds the one-shot all-reduce watchdog (2 s by default), a wedge: the engine stops and leaves the
`rdna_ar_wedged` marker. The tell-tale is the GPU driver's per-process queue-eviction counter rising while
serving:

```sh
for p in $(pgrep -f 'VLLM::Worker_TP'); do echo "$p $(cat /sys/class/kfd/kfd/proc/$p/stats_*/evicted_ms | head -1)"; done
```

A few seconds accumulated during startup is normal. Growth while serving means pages the GPU has mapped are
being moved.

**Mechanism.** The RAM tier (e.g. 12 GiB per rank) and ROCm's own host buffers (~2.5 GB per worker) are
registered with the GPU driver as *userptr* memory: the GPU reads and writes them directly over PCIe, but the
pages remain ordinary process memory that the kernel may move. Memory compaction moves pages to assemble
contiguous free blocks. When it moves one page of a registered range, the driver stops all GPU queues of that
process and re-faults and re-maps the **entire** registration before resuming; for 12 GB that took 5–20 s.
Nothing is corrupted; it costs time, and in tensor parallelism one stalled rank stalls all of them.

**Fix, two parts:**

1. The fork locks that memory (`mlock` of the tier, `mlockall(MCL_ONFAULT)` in each GPU worker;
   `VLLM_RDNA_OFFLOAD_MLOCK`, default on). This needs a locked-memory limit of at least the tier per rank
   plus ~3 GB: set it unlimited. Under systemd the limit is inherited from the user manager
   (`user@<uid>.service`, default hard limit 8 MB) and only root can raise it there, hence the drop-ins.
   Containers: `docker run --ulimit memlock=-1`.
2. `vm.compact_unevictable_allowed = 0`. **The name reads backward:** it answers "may compaction move
   unevictable (mlocked) pages?". The kernel default, 1, lets compaction move locked memory: `mlock`
   guarantees a page stays in RAM, not that it stays at the same physical address. 0 makes compaction skip
   it. This is a host setting; a container cannot set it.

Both are required: the sysctl alone protects nothing (no memory is locked), the lock alone does not stop
compaction.

**Verify:** the serve log contains `CPU offload tier mlocked` and `GPU worker memory mlocked (on fault)` from each
worker. `mlock of the CPU offload tier failed (... RLIMIT_MEMLOCK too low?)` means part 1 is missing;
`... mlocked (but vm.compact_unevictable_allowed=1 ...)` means part 2 is missing. Serving works either way,
with the stalls.

**Cost.** Locked pages cannot be moved, so compaction cannot merge free space across them. The locked memory is a
few large arrays allocated once at startup that never grow or move, so this is fixed at boot (visible as how
much of the tier got 2 MB pages: `CPU offload tensor ... % on 2 MB pages`) and does not accumulate. All other
memory is compacted as before. Locked memory is also never reclaimed: keep the host's memory budget explicit
(tier + n-gram table + workers, with a margin in `MemAvailable`).

**Not needed: `vm.compaction_proactiveness`.** Setting it to 0 turns off background defragmentation. It was the
largest source of page moves before the memory was locked; with both parts above in place it should no
longer reach the vLLM workers. Leave it at the default (20) unless the eviction counters above grow while
serving, or another ROCm *compute* process (HIP; Vulkan programs are not affected) shows queue evictions.

**Measured** (4× V620, 48 GiB tier, 2026-10-01/02): ~120 s of queue eviction per worker over one test run and
13–19 s steps with neither part; with both parts (and `compaction_proactiveness` at 0 during that test), ~25 ms
over a soak plus a 10 GB memory-pressure test. The combination recommended here, with `compaction_proactiveness`
at its default, has not yet been soaked: watch the counters after deploying it. Details in
[`../../docs/rdna2/CHANGES.md`](../../docs/rdna2/CHANGES.md) §19b.
