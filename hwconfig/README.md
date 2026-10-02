# hwconfig — host-side configuration

What the host needs beyond this vLLM fork to run the cards at the operating point in
[`PRODUCTION.md`](../PRODUCTION.md).

| Directory | Contents |
|---|---|
| [`kernel-patches/`](kernel-patches/) | Two amdgpu patches: OverDrive for the Radeon PRO V620, and a 100 W minimum power cap for the V620 and Navi 22. How to apply, build and verify. |
| [`cardinit/`](cardinit/) | Scripts that set the power caps, the V620 clock ceiling and the undervolt after every boot. User-editable values plus a root-owned helper. |

The kernel command line we use is listed in `PRODUCTION.md`.

## Memory settings for the CPU KV-cache tier

Only needed with `--kv-offloading-size` (CHANGES §19b). The GPU driver maps the pinned RAM tier as a *userptr*: if
the kernel moves one of its pages (memory compaction), every GPU queue of that worker stops until the driver has
re-mapped the range, which held a rank for 5–20 s and desynchronised tensor parallelism. The fork locks the tier and
the worker's memory so compaction leaves it alone; that needs:

```sh
# /etc/sysctl.d/90-vllm-offload.conf
vm.compact_unevictable_allowed = 0   # compaction skips mlocked pages
vm.compaction_proactiveness = 0      # no background compaction (largest source of migrations)
```

and an unlimited (or ≥ tier per rank + ~3 GB) locked-memory limit for the vLLM process, e.g. `LimitMEMLOCK=infinity`
in its systemd unit, or `memlock` in `/etc/security/limits.d/` for the user that starts it (new login needed). Check
the serve log for `CPU offload tier mlocked` and `GPU worker memory mlocked`; a warning there means the limit is too low.
