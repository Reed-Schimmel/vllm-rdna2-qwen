#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build only the imported CPU/Linux extension. No torch, ROCm or GPU compiler.
set -euo pipefail

: "${PLE_NVME_SOURCE:?set PLE_NVME_SOURCE to the recovered plehip/csrc/ple_nvme.cpp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Create this worktree's .venv with uv before building" >&2
  exit 1
fi
if [ ! -f "$PLE_NVME_SOURCE" ]; then
  echo "PLE_NVME_SOURCE does not name a source file" >&2
  exit 1
fi
EXPECTED="${PLE_NVME_SOURCE_SHA256:-7f42d0fc4274e8d972a94c39b038c6b2d3217a5b081536b50d553faec0d5b2e6}"
ACTUAL="$(sha256sum "$PLE_NVME_SOURCE" | cut -d' ' -f1)"
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "NVMe source differs from the inspected image; review before overriding PLE_NVME_SOURCE_SHA256" >&2
  exit 1
fi
INCLUDES=()
read -r -a INCLUDES <<< "$("$PYTHON" -m pybind11 --includes)"
SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
OUTPUT="$SITE/ple_nvme_ext$SUFFIX"
TEMP="$(mktemp "$SITE/ple_nvme_ext.XXXXXXXX.so")"
trap 'rm -f "$TEMP"' EXIT
"${CXX:-c++}" -O2 -shared -std=c++17 -fPIC -pthread \
  "${INCLUDES[@]}" "$PLE_NVME_SOURCE" -o "$TEMP"
mv "$TEMP" "$OUTPUT"
echo "Built CPU NVMe extension: $OUTPUT"
