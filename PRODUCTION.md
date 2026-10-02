# Production setup

How this fork runs in production on four Radeon PRO V620s (tensor parallel, TP=4) serving
Qwen3.8-Flash-Next, and the balance of performance, stability and power cost behind that setup.

**Setting up or rebuilding a host? Start at [`hwconfig/`](hwconfig/README.md).** It lists every piece of host
configuration (kernel patches, command line, per-boot GPU settings, OS settings) with where each is made
persistent, and [`hwconfig/check-host.sh`](hwconfig/check-host.sh) checks a running host against it.

## Performance vs. stability

Most of this fork's work went into speed: kernels written for gfx1030, int8 and int4 compute paths, a
custom all-reduce, and faster prefill. Each step moved more work through the cards, and more data between
them over PCIe, in less time. Four cards in tensor parallelism are also tightly coordinated: at every
layer they compute, exchange results and resume together.

As throughput and traffic density rose, stability fell. Configurations that were fine at lower
throughput began to fail under sustained heavy load. So stability became part of the tuning process
rather than something to assume. What mattered most:

- **Steady card power, not just less of it.** A power cap alone makes each card's firmware constantly
  adjust clock and voltage to hold the average, and all four cards do it at the same moments. A clock
  ceiling plus a small undervolt keeps clocks flat, at a more efficient point on the voltage–frequency
  curve, wherever the ceiling rather than the cap is the limit. That held up where plain caps at similar
  or higher power did not, and it is faster than a plain cap at the same electricity cost.
- **The shape of inter-GPU traffic.** Moving data as fewer, larger transactions (`VLLM_RDNA_AR_MODE=wide`,
  `NCCL_PROTO=Simple`), and staggering exchanges so each card talks to one peer at a time, reduced stress
  on the PCIe fabric. It helped, but did not fix the problem on its own.
- **A conservative platform.** No memory overclock, PCIe link power management off, links capped at the
  platform's generation.

The operating point below is the result. It was chosen empirically: it passed repeated sustained load
tests where more aggressive settings failed, and it sits within a few percent of the fastest stable
performance we found. Margins found this way can move with kernel, driver, firmware or model changes, so
we re-run the load test after any of them.

## Operating point

**GPUs.** Applied after every boot by [`hwconfig/cardinit/`](hwconfig/cardinit/); requires the patched
driver in [`hwconfig/kernel-patches/`](hwconfig/kernel-patches/).

| Setting | Value | Notes |
|---|---|---|
| Power cap | **140 W** per card | Down from the 250 W default; needs the 100 W-floor patch. |
| Clock ceiling | **2300 MHz** (V620) | Decode draws less power than prefill, so decode runs flat at the ceiling, below the cap. Heavy prefill reaches the cap and runs a little below the ceiling. |
| Core voltage offset | **−25 mV** (V620) | Undervolt; lowers power at a given clock. Validate model quality after any change. |
| Performance level | `auto` | Idle clocks, voltage and power stay low (~10 W per card). The ceiling only applies under load. |

**Choosing the ceiling.** Decode on this platform is latency-bound: each token runs many small kernels
and about 95 all-reduces, so it scales with GPU clock. Single-stream decode at a 140 W cap:

| Ceiling | Decode |
|---|---|
| 2100 MHz | 53–54 tokens/s |
| 2250 MHz | ~58 tokens/s |
| **2300 MHz** | **~59.5 tokens/s** |

At 2100 MHz the ceiling is below what the cards sustain at the cap even in prefill, so clocks stay flat
everywhere; that is the most conservative setting, and it passed three consecutive load tests. At 2300 MHz
decode runs at the full ceiling, while heavy prefill runs at the cap. There each card settles at about
2050–2160 MHz, depending on its own silicon, and the cap adjusts clocks again. 2300 MHz also passed three
consecutive load tests, so we run it for the decode speed. **If stability suffers, step the ceiling back
down (2250, then 2100) before touching anything else.**

**Host.**

| Setting | Value |
|---|---|
| Memory | Not overclocked. The XMP profile's timings at a reduced speed (2400 MT/s); no XMP frequency. |
| Kernel | Mainline 7.2.6 with the two amdgpu patches in `hwconfig/kernel-patches/`. |
| ROCm | TheRock ROCm 7.14 (see [`docs/rdna2/README.md`](docs/rdna2/README.md)), with the patched ROCr runtime from [`docs/rdna2/ROCR-CPU-FIX.md`](docs/rdna2/ROCR-CPU-FIX.md) preloaded. |

## vLLM configuration

Served with [`tools/rdna2/serve-qwen38-flash-next.sh`](tools/rdna2/serve-qwen38-flash-next.sh). Every
variable is explained in [`docs/rdna2/ENVIRONMENT.md`](docs/rdna2/ENVIRONMENT.md).

| Setting | Value | Purpose |
|---|---|---|
| `MTP` | `0` | No speculative decoding. |
| `MAXLEN` | `262144` | Full model context. |
| `GPUUTIL` | `0.93` | VRAM fraction for vLLM; KV cache takes the rest (~510–525k tokens). |
| `DENSE_INT8`, `DENSE_INT8_ONLY` | `1`, `1` | int8 copies of the dense projections for decode, and the fp16 copies freed (more KV cache). |
| `VISION` | `1` | Image input enabled. |
| `PLE_INT4` | int4 n-gram table sidecar (~32 GB) | The n-gram table, served from host memory by the CPU offload worker. The fp8 sidecar (52.5 GB) is ~3× cleaner per row, but measured gains were small (general text within noise; −0.41 % NLL on identifier/digit-heavy text, ~1.5 SE); int4 keeps ~20 GB more RAM free. |
| `EXTRA_ARGS` | `--prefix-cache-retention-interval 6272 --max-num-seqs 2 --kv-offloading-size 48` | Retain linear-attention state every 8 blocks along long prompts, for faster follow-up turns; at most 2 requests run at once (the rest queue); 48 GiB of CPU RAM as a second KV-cache tier (~960k tokens), so conversations evicted from the GPU reload in seconds instead of being recomputed. |
| `VLLM_USE_SIMPLE_KV_OFFLOAD` | `1` | Required with this model for the CPU tier (see [`docs/rdna2/CHANGES.md`](docs/rdna2/CHANGES.md) §19). |
| `PLE_OFFLOAD_ANON` | `1` | The n-gram worker keeps the 52.5 GB table in its own memory, not the page cache, so memory pressure can't evict it. Required with the CPU tier: without it, evicted table pages made lookups take seconds and desynchronised the cards. |
| `VLLM_RDNA_AR` | `1` | Custom one-shot all-reduce for decode-sized messages. |
| `VLLM_RDNA_AR_MODE` | `wide` | 16-byte writes forming whole 128-byte lines, writes only, local waiting. The serve script's default. |
| `VLLM_RDNA_AR_BLOCKS`, `VLLM_RDNA_AR_PACE` | `4`, `16` | Fewer concurrent write streams, with spacing between bursts. |
| `VLLM_RDNA_AR_Q8` | `1` | int8-compressed prefill all-reduce, staggered exchange (the default). |
| `VLLM_RDNA_MOE_W4A8` | `1` | int8-activation × int4-expert MoE kernel for prefill. |
| `VLLM_RDNA_DENSE_W8A8` | `1` | int8 × int8 prefill GEMMs for the dense projections. |
| `NCCL_PROTO` | `Simple` | RCCL moves data in large chunks rather than flagged 8-byte stores. The serve script's default. |
| `NCCL_P2P_LEVEL`, `NCCL_GRAPH_MIXING_SUPPORT` | `SYS`, `1` | Set by the serve script: direct card-to-card RCCL, and correct graph and eager mixing. |
| TunableOp | lookup-only | Tuned GEMM rows for the installed rocBLAS build; never tuned while serving. |

**Host memory budget.** With the CPU tier, RAM is committed explicitly: the pinned tier, the resident n-gram table,
~6 GB per GPU worker, plus whatever else the host runs. 64 GiB of tier with the fp8 table left only ~15 GB of the
188 GB available, so production runs 48 GiB with the int4 table (~32 GB). Size `--kv-offloading-size` so
`MemAvailable` keeps a margin for the rest of the system.

Single-stream decode is about 59.5 tokens/s at the production operating point; the published container, with its
defaults, reaches about 64 tokens/s on the same cards. The likeliest contributor is the full **262k context** (the
container uses 131k): decode runs as captured CUDA graphs, so the sparse-attention indexer scores against the full
page-table capacity, which scales with the maximum context length. The table format is not the cause: measured
09-16, a warm fp8 lookup was cheaper than int4 (0.08–0.14 vs 0.24–0.34 ms) and fp8 decode was 63.7–64.0 vs int4
62.6 t/s in the same session. Not separately measured: vision on, int8-only weights, and the CPU tier. Prefill throughput depends on prompt length and concurrency, averaging about
1,150–1,250 tokens/s under our mixed multi-request load test.

## Kernel command line

The arguments that relate to the AMD GPUs and to inference:

| Argument | Purpose |
|---|---|
| `amd_iommu=on iommu=pt` | IOMMU in passthrough: device DMA, including card-to-card traffic, is not translated. (`amd_iommu=on` isn't a recognised value and is ignored; `iommu=pt` is what takes effect.) |
| `pci=realloc=off` | Keep the firmware's PCI resource assignment. The cards' full-size BARs (32 GiB) are already mapped. |
| `pcie_aspm=off`, `amdgpu.aspm=0` | No PCIe link power states: links stay fully active. |
| `pcie_port_pm=off` | No runtime power management on PCIe ports. |
| `amdgpu.runpm=0` | No runtime power-down of idle GPUs. |
| `amdgpu.pcie_gen_cap=0x00070007` | Allow PCIe Gen1–Gen3 only, both for the card and the platform. Matches the slots; also the knob for a Gen2 test (`0x00030003`). |
| `amdgpu.ppfeaturemask=0xfff77fff` | Power-play feature mask: **OverDrive on** (`0x4000`, needed for the clock ceiling and undervolt) and **GFXOFF off** (`0x8000` cleared), avoiding power-gating transitions under bursty load. |
| `amdgpu.gpu_recovery=1` | Allow the driver to attempt GPU recovery after a hang. |
| `amdgpu.noretry=0` | Retry on GPU page faults instead of failing fast. The minimal line in `docs/rdna2/README.md` uses `amdgpu.noretry=1`, the more conservative choice; we haven't isolated a difference between the two in production. |
| `amdgpu.ras_enable=1` | Enable RAS error reporting where the hardware supports it. |
| `amdgpu.gartsize=4096` | 4 GiB GART, for GPU-visible system memory. |
| `ttm.pages_limit=16777216`, `ttm.page_pool_size=1048576` | Allow up to 64 GiB of GPU-mapped system memory (GTT), with a 4 GiB page pool, for large pinned host buffers. |

Not needed and removable: `pci=earlydump` (debug output) and `amdttm.*` (only read by AMD's out-of-tree
DKMS driver; with the in-kernel driver the `ttm.*` equivalents apply). `pci=nomsi` should **not** be used:
it forces every device onto legacy shared interrupts.

## Validation

Before adopting a change we run a 4-minute load test. It uses four concurrent workers mixing fresh long
prompts (2k–40k tokens), follow-up turns and decodes, while monitoring kernel events and each card's
power, clocks, voltage and temperature. Both the 2100 MHz and the 2300 MHz operating points passed three
consecutive runs with no errors and no kernel events. We treat that as strong evidence, not a guarantee: confidence grows with
accumulated clean runtime at a fixed configuration, and we re-test after kernel, driver, firmware, BIOS or
model changes.
