#!/usr/bin/env bash
# Read-only check of everything this fork expects from the host (hwconfig/README.md). No root needed.
# Prints one line per item: OK, FAIL (fix it), WARN (check it) or INFO, each with where the fix is documented.
#
#   hwconfig/check-host.sh                       # targets from hwconfig/cardinit/cardinit
#   hwconfig/check-host.sh --cardinit FILE       # targets from your edited copy of cardinit
#   hwconfig/check-host.sh --unit NAME           # also check the memlock limit of systemd user unit NAME
#   hwconfig/check-host.sh --cooling PROCESS     # also check that a cooling/fan-control process is running
#   hwconfig/check-host.sh --no-offload          # the CPU KV tier is not used: memory items become INFO
#   V620_POWER=140 V620_SCLK=2300 hwconfig/check-host.sh   # override single targets (as with cardinit arguments)
#
# Exit status: number of FAIL lines (0 = all good).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CARDINIT="$HERE/cardinit/cardinit"; UNIT=""; COOLING=""; OFFLOAD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --cardinit) CARDINIT="$2"; shift 2 ;;
    --unit) UNIT="$2"; shift 2 ;;
    --cooling) COOLING="$2"; shift 2 ;;
    --no-offload) OFFLOAD=0; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

FAILS=0
ok()   { printf '  OK    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }
warn() { printf '  WARN  %s\n' "$*"; }
info() { printf '  INFO  %s\n' "$*"; }
mem_fail() { if [ "$OFFLOAD" = 1 ]; then fail "$*"; else info "$* (only needed with --kv-offloading-size)"; fi; }

# ---------------------------------------------------------------------------------------------------------------
echo "== kernel and driver   (hwconfig/kernel-patches/, PRODUCTION.md 'Kernel command line')"
info "kernel $(uname -r), amdgpu module $(modinfo -F filename amdgpu 2>/dev/null || echo '?')"
CMDLINE=" $(cat /proc/cmdline) "
for arg in iommu=pt pcie_aspm=off amdgpu.aspm=0 amdgpu.runpm=0 pcie_port_pm=off amdgpu.pcie_gen_cap= \
           amdgpu.ppfeaturemask= amdgpu.gartsize= ttm.pages_limit= ttm.page_pool_size=; do
  if [[ "$CMDLINE" == *" $arg"* ]]; then ok "cmdline has $arg"; else fail "cmdline lacks $arg"; fi
done
mask=$(grep -o 'amdgpu.ppfeaturemask=0x[0-9a-fA-F]*' /proc/cmdline | cut -d= -f2)
if [ -n "$mask" ]; then
  if (( mask & 0x4000 )); then ok "ppfeaturemask $mask has OverDrive (0x4000)"; else fail "ppfeaturemask $mask lacks OverDrive (0x4000): no clock ceiling or undervolt"; fi
  if (( mask & 0x8000 )); then warn "ppfeaturemask $mask has GFXOFF (0x8000) enabled; production clears it"; else ok "ppfeaturemask clears GFXOFF (0x8000)"; fi
fi
[[ "$CMDLINE" == *" pci=nomsi"* ]] && fail "cmdline has pci=nomsi: forces legacy shared interrupts, remove it"

# ---------------------------------------------------------------------------------------------------------------
echo "== GPUs   (hwconfig/cardinit/: run cardinit after every boot)"
tv() { local v="${!1:-}"; [ -n "$v" ] && { echo "$v"; return; }; grep -m1 "^$1=" "$CARDINIT" 2>/dev/null | cut -d= -f2 | awk '{print $1}'; }
T_POWER=$(tv V620_POWER); T_OTHER=$(tv OTHER_POWER); T_SCLK=$(tv V620_SCLK); T_VOFF=$(tv V620_VOFFSET)
info "targets (env overrides, else $CARDINIT): V620 ${T_POWER:-?} W / ${T_SCLK:-?} MHz / ${T_VOFF:-?} mV, other GPUs ${T_OTHER:-?} W"
n_v620=0
for d in /sys/bus/pci/drivers/amdgpu/0000:*; do
  [ -f "$d/uevent" ] || continue
  bdf=$(basename "$d"); id=$(grep -m1 PCI_ID "$d/uevent" | cut -d= -f2)
  h=$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -1)
  cap=$(( $(cat "$h/power1_cap" 2>/dev/null || echo 0) / 1000000 ))
  capmin=$(( $(cat "$h/power1_cap_min" 2>/dev/null || echo 0) / 1000000 ))
  if [ "$id" = "1002:73A1" ] || [ "$id" = "1002:73a1" ]; then
    n_v620=$((n_v620 + 1))
    od="$d/pp_od_clk_voltage"
    if [ ! -r "$od" ]; then fail "$bdf V620: no pp_od_clk_voltage (OverDrive patch or ppfeaturemask missing)"; continue; fi
    sclk=$(awk '/^OD_SCLK/{f=1;next} f&&/^ *1:/{gsub(/[^0-9]/,"",$2); print $2; exit}' "$od")
    voff=$(awk '/^OD_VDDGFX_OFFSET/{getline; gsub(/[^0-9-]/,""); print; exit}' "$od")
    msg="$bdf V620: cap ${cap} W, clock ceiling ${sclk} MHz, voltage offset ${voff} mV"
    if [ "$cap" = "${T_POWER:-x}" ] && [ "$sclk" = "${T_SCLK:-x}" ] && [ "$voff" = "${T_VOFF:-x}" ]; then ok "$msg"
    else fail "$msg -- not the cardinit targets (run cardinit)"; fi
    [ "$capmin" -le 100 ] 2>/dev/null || warn "$bdf V620: minimum cap ${capmin} W (100 W-floor patch not loaded?)"
  else
    msg="$bdf $id: cap ${cap} W"
    if [ -n "$T_OTHER" ] && [ "$cap" != "$T_OTHER" ]; then fail "$msg -- cardinit target ${T_OTHER} W (run cardinit)"; else ok "$msg"; fi
  fi
done
[ "$n_v620" -gt 0 ] && info "$n_v620 Radeon PRO V620 on the bus" || fail "no Radeon PRO V620 found on the amdgpu driver"

if [ -n "$COOLING" ]; then
  if pgrep -f "$COOLING" >/dev/null; then ok "cooling process '$COOLING' running"; else fail "cooling process '$COOLING' NOT running: start it before any load"; fi
fi

# ---------------------------------------------------------------------------------------------------------------
echo "== memory for the CPU KV-cache tier   (hwconfig/os/README.md)"
cua=$(cat /proc/sys/vm/compact_unevictable_allowed 2>/dev/null)
if [ "$cua" = 0 ]; then ok "vm.compact_unevictable_allowed = 0 (compaction leaves mlocked memory alone)"
else mem_fail "vm.compact_unevictable_allowed = $cua: compaction may move the locked RAM tier -> multi-second rank stalls (install os/etc/sysctl.d/90-vllm-offload.conf)"; fi
info "vm.compaction_proactiveness = $(cat /proc/sys/vm/compaction_proactiveness) (default 20 is fine; see os/README.md)"
thp=$(grep -o '\[[a-z]*\]' /sys/kernel/mm/transparent_hugepage/enabled | tr -d '[]')
if [ "$thp" = never ]; then warn "transparent hugepages 'never': the RAM tier falls back to 4 KB pages"; else ok "transparent hugepages '$thp'"; fi

lim() { awk '/Max locked memory/{print $4}' "/proc/$1/limits" 2>/dev/null; }
um=$(pgrep -u "$(id -u)" -x systemd | head -1)
if [ -n "$um" ]; then
  l=$(lim "$um")
  if [ "$l" = unlimited ]; then ok "user manager (systemd --user) memlock limit unlimited"
  else mem_fail "user manager memlock limit $l bytes: units and their children cannot lock the RAM tier (install os/etc/systemd/system/user@.service.d/memlock.conf, then reboot)"; fi
fi
if [ -n "$UNIT" ]; then
  l=$(systemctl --user show "$UNIT" -p LimitMEMLOCK --value 2>/dev/null)
  if [ "$l" = infinity ]; then ok "unit $UNIT LimitMEMLOCK=infinity"
  else mem_fail "unit $UNIT LimitMEMLOCK=${l:-?} (install os/systemd-user/vllm.service.d/memlock.conf as $UNIT.d/)"; fi
fi
info "this shell: ulimit -l $(ulimit -l)"

# ---------------------------------------------------------------------------------------------------------------
echo "== running vLLM   (only if a server is up)"
workers=$(pgrep -f 'VLLM::Worker_TP' || true)
if [ -z "$workers" ]; then info "no vLLM GPU workers running"
else
  for p in $workers; do
    ev=$(cat /sys/class/kfd/kfd/proc/"$p"/stats_*/evicted_ms 2>/dev/null | head -1)
    info "worker $p: memlock $(lim "$p"), queue eviction so far ${ev:-?} ms (should not grow while serving)"
  done
fi
marker="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/rdna_ar_wedged"
if [ -e "$marker" ]; then warn "$marker exists: next start falls back to RCCL (CHANGES.md: one-shot all-reduce watchdog)"
else ok "no rdna_ar_wedged marker in ${marker%/*} (set VLLM_CACHE_ROOT to check another cache root)"; fi

echo
[ "$FAILS" -eq 0 ] && echo "all checks passed" || echo "$FAILS check(s) failed"
exit "$FAILS"
