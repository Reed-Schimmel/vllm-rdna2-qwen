# hwconfig — host-side configuration

Everything the host needs beyond this vLLM fork, in one place. **After rebuilding or reprovisioning a machine,
go down this table, then run [`check-host.sh`](check-host.sh)**: it checks every row that can be checked from a
running system and points at the fix for each failure.

```sh
hwconfig/check-host.sh --cardinit <your cardinit copy> --unit <user unit that starts vLLM> --cooling <fan process>
```

| What | Source of truth | Made persistent by | Needed for | If missing |
|---|---|---|---|---|
| Kernel with two amdgpu patches | [`kernel-patches/`](kernel-patches/) | your kernel build / package | V620 OverDrive (clock ceiling, undervolt) and power caps below 250 W | `cardinit` cannot set the operating point; no `pp_od_clk_voltage` on the V620s |
| Kernel command line | [`../PRODUCTION.md`](../PRODUCTION.md#kernel-command-line) | bootloader config (e.g. `GRUB_CMDLINE_LINUX` in `/etc/default/grub`, then `update-grub`) | OverDrive bit, link power management off, PCIe gen cap, GTT size for large pinned buffers | missing OverDrive, unstable links under load, pinned allocations failing |
| GPU operating point (power cap, clock ceiling, undervolt) | [`cardinit/`](cardinit/) (edit the values in your copy) | **nothing by default: run `cardinit` after every boot**, or a oneshot unit ([`cardinit/README.md`](cardinit/README.md)) | stability under sustained tensor-parallel load ([`../PRODUCTION.md`](../PRODUCTION.md#operating-point)) | cards at 250 W stock limits and boost clocks |
| GPU cooling | site-specific (fan controller) | site-specific | any load | thermal throttling, unreliable timings |
| `vm.compact_unevictable_allowed = 0` | [`os/etc/sysctl.d/90-vllm-offload.conf`](os/etc/sysctl.d/90-vllm-offload.conf) | `/etc/sysctl.d/` | CPU KV-cache tier (`--kv-offloading-size`) | 5–20 s stalls on all ranks, `PLE lookup … >5 s`, possible all-reduce wedge ([`os/README.md`](os/README.md)) |
| Locked-memory limit unlimited | [`os/etc/systemd/system/user@.service.d/`](os/etc/systemd/system/user@.service.d/), [`os/systemd-user/`](os/systemd-user/) | systemd drop-ins (reboot to apply) | CPU KV-cache tier | serve log `mlock … failed`; same stalls as above |
| Transparent hugepages `madvise` or `always` | distribution default | — | CPU KV-cache tier (2 MB pages) | tier on 4 KB pages (more compaction exposure) |

Containers: the device, IPC and memlock flags are in [`../containers/README.md`](../containers/README.md)
(`--ulimit memlock=-1`). Kernel, command line, operating point and the sysctl are host settings that no
container flag can replace.

| Directory | Contents |
|---|---|
| [`kernel-patches/`](kernel-patches/) | Two amdgpu patches: OverDrive for the Radeon PRO V620, and a 100 W minimum power cap for the V620 and Navi 22. How to apply, build and verify. |
| [`cardinit/`](cardinit/) | Scripts that set the power caps, the V620 clock ceiling and the undervolt after every boot. User-editable values plus a root-owned helper. |
| [`os/`](os/) | OS settings as files laid out like their target paths (sysctl, systemd drop-ins), with the reasoning and the symptoms of each being missing. |
| [`check-host.sh`](check-host.sh) | Read-only check of all of the above on a running host. |
