#!/usr/bin/env bash
# Spin up the Qwen3.8-Flash-Next server on 4x Radeon PRO V620 (gfx1030) with this fork's
# recommended runtime settings and Qwen's official Thinking-mode sampling defaults.
#
# This is a thin, self-contained wrapper around tools/rdna2/serve-qwen38-flash-next.sh:
#   - activates the prebuilt venv
#   - points at the local model + PLE sidecar
#   - selects the four physical GPUs present on this host (ROCR ids 0,1,2,3)
#   - sets server-side DEFAULT sampling params (clients may still override per request)
#   - sets server-side DEFAULT thinking behaviour (clients may still override per request)
#
# Server-side default sampling params (Qwen3.8-Flash-Next "Thinking Mode" recommendation,
# https://huggingface.co/Qwen/Qwen3.8-Flash-Next -> Best Practices):
#   temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0
#   presence_penalty=0.0 (vLLM's built-in default; there is no server flag for it, and the
#     model's recommended value is 0.0, so it is already in effect without being set here)
# These become the defaults for any request that does not send its own value; a client that
# sends temperature/top_p/etc. overrides them for that request.
#
# Server-side default thinking behaviour (chat_template_kwargs; request values win):
#   enable_thinking=true, preserve_thinking=true, reasoning_effort=medium
#
# Run directly:
#   tools/rdna2/start-qwen38-service.sh
# Or as the ExecStart of the systemd unit tools/rdna2/qwen38-flash-next.service.
#
# Overridable via the environment (sensible defaults for THIS host are baked in):
#   REPO_ROOT, VENV, MODEL, PLE_INT4, GPUS, PORT, plus every knob of
#   serve-qwen38-flash-next.sh (MTP, GPUUTIL, MAXLEN, TOOLS, VISION, ...).
set -euo pipefail

# --- locations (defaults resolved relative to this script) --------------------------------
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$_SCRIPT_DIR/../.." && pwd)}"
VENV="${VENV:-$HOME/venvs/vllm-rdna2-qwen}"
MODEL="${MODEL:-$REPO_ROOT/models/qwen38-flash-next}"
PLE_INT4="${PLE_INT4:-$REPO_ROOT/models/qwen38-flash-next-ple/ples_int4}"

# --- GPUs: this host has exactly four gfx1030 cards at ROCR ids 0,1,2,3 --------------------
# (the serve script's own default of 1,2,3,4 is wrong here: index 4 does not exist)
export GPUS="${GPUS:-0,1,2,3}"
export PORT="${PORT:-8000}"

# --- KV cache dtype ------------------------------------------------------------------------
# The KV cache stays fp16. The int8 features below quantise WEIGHTS (and prefill activations /
# all-reduce traffic), never the KV cache. fp8 KV is not an option anyway: the model's sparse
# attention (vllm/models/qwen4_exp/amd/qsa.py) accepts auto or bfloat16 and rejects explicit
# float16 and fp8. Use auto with --dtype float16 (the default here).
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# --- context length / VRAM utilization -----------------------------------------------------
# Headless host (~15 MB/GPU idle). The docs advise GPUUTIL <= 0.93 with DENSE_INT8_ONLY=1
# (CHANGES.md #7, ~2 GiB idle headroom); this host runs 0.95 to grow the KV pool. Drop back to
# 0.93 on OOM. vLLM's CUDA-graph memory profiling makes 0.95 equivalent to ~0.90 without it.
# MAXLEN is the model's full context (max_position_embeddings = 262144).
# Measured KV pool, MAXLEN 262144, int8 features on, vision tower on CPU (2026-10-04/05):
#   GPUUTIL 0.93, MTP=3, 4 seqs: 473,962 tokens (1.81x a full-length request)
#   GPUUTIL 0.95, MTP=0, 8 seqs: 721,658 tokens (2.75x)   <- these defaults
export GPUUTIL="${GPUUTIL:-0.95}"
export MAXLEN="${MAXLEN:-262144}"
# MTP speculative decoding off: frees its draft weights/KV for the main KV pool at the cost
# of single-stream decode speed (~80 -> ~64 t/s); aggregate throughput at 8 concurrent
# streams reached ~185-200 t/s. MTP=3 restores it.
export MTP="${MTP:-0}"
# Concurrent sequences. The fork's decode kernels cover batches <= 8 tokens (rdna_ops.py
# _DECODE_MAX); above that decode drops to the prefill path. Measured 2026-10-05, 256-token
# streams: 8 running -> ~185 t/s aggregate, 16 running -> ~130 t/s (9 t/s per stream).
# Keep at 8 (with MTP>0 each sequence decodes 1+MTP tokens per step: use 2 for MTP=3).
export MAX_SEQS="${MAX_SEQS:-8}"

# --- int8 performance features (docs/rdna2/CHANGES.md #7, #10, #13, #14, #17) --------------
# These affect weights, prefill activations and the prefill all-reduce -- not the KV cache.
# DENSE_INT8: int8 weight shadows for decode GEMVs. DENSE_INT8_ONLY: drop the fp16 weight
# copies (~2-3 GiB/card back to the KV pool; prefill dequantises the shadows).
# DENSE_W8A8: int8 x int8 prefill GEMMs for the dense projections (needs DENSE_INT8=1).
# MOE_W4A8: int8-activation x int4-expert MoE prefill kernel (~+27 % prefill).
# AR_Q8: int8-compressed prefill all-reduce (staggered exchange is on by default).
export DENSE_INT8="${DENSE_INT8:-1}"
export DENSE_INT8_ONLY="${DENSE_INT8_ONLY:-1}"
export VLLM_RDNA_DENSE_W8A8="${VLLM_RDNA_DENSE_W8A8:-1}"
export VLLM_RDNA_MOE_W4A8="${VLLM_RDNA_MOE_W4A8:-1}"
export VLLM_RDNA_AR_Q8="${VLLM_RDNA_AR_Q8:-1}"
# DENSE_INT8_ONLY changes the compiled graph: keep its compile/cache artifacts separate.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm-int8only}"

# --- vision ---------------------------------------------------------------------------------
# Load the model's Qwen3-VL-style vision tower on every rank and accept images on the chat
# API. Costs ~0.9 GB/card for the tower plus KV pool for image tokens; the serve script keeps
# mm-profiling ON when VISION=1 so the encoder's activations are accounted for before the KV
# pool is sized. Image limit and resize caps come from MM_LIMIT / MM_PROCESSOR_KWARGS
# (defaults: 4 images/prompt, resized to <=1280x1280). Do not raise max_pixels casually: on
# gfx1030 the ViT runs SDPA's math path (materializes NxN), and a 16 MP image asks ~64 GiB.
export VISION="${VISION:-1}"

# Offload the vision tower's weights (~0.9 GB/card) to pinned CPU RAM via UVA (zero-copy),
# freeing that VRAM for the KV pool. The tower is only touched on image prefill (text decode
# is unaffected); the cost is PCIe bandwidth when processing images. Value is the GiB/card
# budget; 2 comfortably covers the tower. Set to 0 to keep the tower resident on the GPU.
export VISION_CPU_OFFLOAD="${VISION_CPU_OFFLOAD:-2}"

# --- server-side DEFAULT sampling params (clients override per request) -------------------
# NOTE: EXTRA_ARGS is injected UNQUOTED by serve-qwen38-flash-next.sh, so this JSON must not
# contain spaces. presence_penalty is intentionally omitted: it is not carried by
# --override-generation-config and its recommended value (0.0) is already vLLM's default.
#
# NOTE: plain assignment, NOT ${EXTRA_ARGS:-...}: bash terminates a ${VAR:-default} word at an
# unescaped '}', which silently eats/shifts JSON closing braces (the CHAT_KWARGS line below
# shows the doubled-brace workaround). The if-form keeps the JSON literal and unambiguous.
if [ -z "${EXTRA_ARGS:-}" ]; then
  EXTRA_ARGS='--override-generation-config {"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0}'
fi

# --- CPU KV-cache tier (docs/rdna2/CHANGES.md #19, #19b) -------------------------------------
# A second KV tier in pinned RAM: prefixes evicted from the GPU pool reload in ~1-2 s instead of
# being recomputed (tens of seconds to minutes for long contexts). It does NOT raise how much
# context can be *active* at once (that is still the GPU pool); it raises how much can be
# *resumed* cheaply. Lazy mode copies a block only as it leaves the GPU, so resumable context is
# ~GPU pool + tier. KV_OFFLOAD_GB is GiB in total across the 4 ranks (~54 KB/token summed over
# ranks: 32 GiB ~ 660k tokens). KV_OFFLOAD_GB=0 disables it.
#   - VLLM_USE_SIMPLE_KV_OFFLOAD=1: vLLM's default OffloadingConnector fails at startup on this
#     model (tokens_per_block=4 not divisible by tokens_per_hash=784, QSA compressor ring).
#   - PLE_OFFLOAD_ANON=1: keep the n-gram table in the worker's own memory; otherwise the pinned
#     tier squeezes it out of the page cache and lookups stall for seconds.
#   - Host: unlimited RLIMIT_MEMLOCK (the systemd unit sets LimitMEMLOCK=infinity) and
#     vm.compact_unevictable_allowed=0 (hwconfig/os/etc/sysctl.d/90-vllm-offload.conf). Without
#     them it still serves, but compaction can stall all ranks for 5-20 s (hwconfig/os/README.md).
# RAM budget on this 123 GB host: table ~30 GB + workers/engine ~28 GB + tier 32 GiB leaves ~25 GB.
#   - VLLM_RDNA_OFFLOAD_MLOCK / _THP / _PIN_CHUNK_MB are on by default in the fork (lock the
#     tier, back it with huge pages, register it in ~128 MB chunks); nothing to set here.
#   - vm.compaction_proactiveness: leave at the default (20); not needed once the tier is locked.
export KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-32}"
if [ "$KV_OFFLOAD_GB" != "0" ] && [[ " $EXTRA_ARGS " != *" --kv-offloading-size "* ]]; then
  export VLLM_USE_SIMPLE_KV_OFFLOAD=1
  export PLE_OFFLOAD_ANON="${PLE_OFFLOAD_ANON:-1}"
  export VLLM_RDNA_OFFLOAD_LAZY_TARGET="${VLLM_RDNA_OFFLOAD_LAZY_TARGET:-96}"
  EXTRA_ARGS+=" --kv-offloading-size $KV_OFFLOAD_GB"
  EXTRA_ARGS+=' --kv-transfer-config {"kv_connector_extra_config":{"lazy_offload":true}}'
fi

# Linear-attention (Mamba) state checkpoints every RETENTION tokens along long prompts (8 Mamba
# blocks of 784), so a follow-up turn or branched conversation resumes from the nearest
# checkpoint instead of recomputing from the start; pairs with the CPU tier (production config,
# PRODUCTION.md). Costs a few KV blocks per long prompt. RETENTION=0 = only the prompt's end.
export RETENTION="${RETENTION:-6272}"
if [[ " $EXTRA_ARGS " != *" --prefix-cache-retention-interval "* ]]; then
  EXTRA_ARGS+=" --prefix-cache-retention-interval $RETENTION"
fi
export EXTRA_ARGS

# --- server-side DEFAULT thinking behaviour (clients override per request) -----------------
export CHAT_KWARGS="${CHAT_KWARGS:-{\"enable_thinking\": true, \"preserve_thinking\": true, \"reasoning_effort\": \"medium\"}}"

# --- activate the prebuilt gfx1030 venv ----------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: venv python not found at $VENV/bin/python" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$REPO_ROOT"
export MODEL PLE_INT4

echo "Starting Qwen3.8-Flash-Next server:"
echo "  repo:      $REPO_ROOT"
echo "  venv:      $VENV"
echo "  model:     $MODEL"
echo "  ple_int4:  $PLE_INT4"
echo "  gpus:      $GPUS   port: $PORT"
echo "  kv cache:  $KV_CACHE_DTYPE"
echo "  context:   max-model-len=$MAXLEN  gpu-util=$GPUUTIL"
echo "  int8:      DENSE_INT8=$DENSE_INT8 DENSE_INT8_ONLY=$DENSE_INT8_ONLY W8A8=$VLLM_RDNA_DENSE_W8A8 MOE_W4A8=$VLLM_RDNA_MOE_W4A8 AR_Q8=$VLLM_RDNA_AR_Q8"
echo "  cache:     VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "  cpu kv:    KV_OFFLOAD_GB=$KV_OFFLOAD_GB (lazy; 0 = off)  retention=$RETENTION  memlock=$(ulimit -l)  compact_unevictable_allowed=$(cat /proc/sys/vm/compact_unevictable_allowed)"
echo "  vision:    VISION=$VISION  cpu_offload_gb=$VISION_CPU_OFFLOAD (tower->CPU via UVA)"
echo "  sampling:  temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 repetition_penalty=1.0 (presence_penalty=0.0 default)"
echo "  thinking:  enable_thinking=true preserve_thinking=true reasoning_effort=medium"

exec "$_SCRIPT_DIR/serve-qwen38-flash-next.sh"
