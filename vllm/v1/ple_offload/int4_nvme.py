# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless INT4 sidecar storage for the RDNA4 CPU NVMe row-cache ABI."""

import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from vllm.logger import init_logger

logger = init_logger(__name__)

INT4_LAYOUT = "group16_int4_fp16scale_lownibblefirst"
TABLE_NAME = "ple_int4.bin"
MANIFEST_NAME = "ple_int4.json"
GIB = 1 << 30
DIO_ALIGNMENT = 4096


def _identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _source_spec(quant_dir: Path) -> dict:
    if sys.byteorder != "little":
        raise ValueError("INT4 NVMe storage requires a little-endian host")
    meta_bytes = (quant_dir / "META.json").read_bytes()
    meta = json.loads(meta_bytes)
    if meta.get("layout") != INT4_LAYOUT:
        raise ValueError(f"INT4 NVMe requires layout {INT4_LAYOUT!r}")
    for key in ("rows", "width", "shards"):
        if type(meta.get(key)) is not int or meta[key] <= 0:
            raise ValueError(f"Invalid INT4 sidecar {key}")
    if meta["width"] % 16 or meta["rows"] % meta["shards"]:
        raise ValueError("INT4 NVMe requires group-16 width and uniform shards")
    row_bytes = meta["width"] // 2 + meta["width"] // 16 * 2
    return {
        "format": 1,
        "source_dir": str(quant_dir.resolve()),
        "layout": INT4_LAYOUT,
        "rows": meta["rows"],
        "width": meta["width"],
        "shards": meta["shards"],
        "row_bytes": row_bytes,
        "meta_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "source_files": [
            _identity(quant_dir / f"shard_{i}.safetensors")
            for i in range(meta["shards"])
        ],
    }


def _lock(directory: Path) -> int:
    fd = os.open(directory / "ple_int4.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError(f"Another process owns the INT4 NVMe table: {directory}")
    return fd


def _validate_table(directory: Path, spec: dict) -> dict:
    with (directory / MANIFEST_NAME).open() as file:
        manifest = json.load(file)
    if manifest.get("source") != spec:
        raise ValueError("INT4 NVMe source changed; prepare a new table directory")
    expected_size = (
        (spec["rows"] * spec["row_bytes"] + DIO_ALIGNMENT - 1)
        // DIO_ALIGNMENT
        * DIO_ALIGNMENT
    )
    identity = _identity(directory / TABLE_NAME)
    if identity["size"] != expected_size:
        raise ValueError("INT4 NVMe table has an incorrect size")
    if manifest.get("table_identity") != identity:
        raise ValueError("INT4 NVMe table changed after preparation")
    return manifest


def prepare_int4_nvme_table(
    quant_dir: str | Path,
    directory: str | Path,
    chunk_rows: int = 65536,
) -> Path:
    """Interleave packed codes and FP16 scales without requantizing any values.

    Existing matching tables are reused. Existing mismatching tables are never
    overwritten; select a new destination directory to rebuild.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    quant_dir, directory = Path(quant_dir).resolve(), Path(directory).resolve()
    if quant_dir == directory:
        raise ValueError("Use a separate directory for the derived NVMe table")
    spec = _source_spec(quant_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_fd = _lock(directory)
    temporary: Path | None = None
    manifest_tmp: Path | None = None
    try:
        table = directory / TABLE_NAME
        if table.exists() or (directory / MANIFEST_NAME).exists():
            _validate_table(directory, spec)
            return table
        byte_count = spec["rows"] * spec["row_bytes"]
        padded_bytes = (byte_count + DIO_ALIGNMENT - 1) // DIO_ALIGNMENT
        padded_bytes *= DIO_ALIGNMENT
        if shutil.disk_usage(directory).free < padded_bytes + GIB:
            raise RuntimeError("Insufficient space for the INT4 NVMe table")
        fd, name = tempfile.mkstemp(prefix="ple_int4.", suffix=".tmp", dir=directory)
        temporary = Path(name)
        rows_per_shard = spec["rows"] // spec["shards"]
        packed_width = spec["width"] // 2
        scale_width = spec["width"] // 16
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb", buffering=0) as output:
            for shard in range(spec["shards"]):
                path = quant_dir / f"shard_{shard}.safetensors"
                with safe_open(path, framework="pt", device="cpu") as source:
                    packed = source.get_slice("weight_i4")
                    scale = source.get_slice("weight_scale")
                    if (
                        packed.get_shape() != [rows_per_shard, packed_width]
                        or scale.get_shape() != [rows_per_shard, scale_width]
                        or packed.get_dtype() != "U8"
                        or scale.get_dtype() != "F16"
                    ):
                        raise ValueError(f"Invalid INT4 shard shape or dtype: {path}")
                    for lo in range(0, rows_per_shard, chunk_rows):
                        hi = min(lo + chunk_rows, rows_per_shard)
                        q = packed[lo:hi].numpy()
                        s = scale[lo:hi].numpy().view(np.uint8)
                        fused = np.concatenate((q, s), axis=1)
                        view = memoryview(fused).cast("B")
                        digest.update(view)
                        while view:
                            written = output.write(view)
                            if not written:
                                raise OSError("Short write preparing INT4 NVMe table")
                            view = view[written:]
                        del q, s, fused, view
                del packed, scale
                os.fdatasync(output.fileno())
                os.posix_fadvise(
                    output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                )
                # Source pages are not the run-time cache, either.
                with path.open("rb") as source_file:
                    os.posix_fadvise(
                        source_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                    )
            output.truncate(padded_bytes)
            os.fsync(output.fileno())
            os.posix_fadvise(output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if _source_spec(quant_dir) != spec:
            raise ValueError("INT4 source changed during preparation")
        os.replace(temporary, table)
        temporary = None
        manifest = {
            "source": spec,
            "table_identity": _identity(table),
            "row_data_sha256": digest.hexdigest(),
        }
        fd, name = tempfile.mkstemp(
            prefix="ple_int4.", suffix=".json.tmp", dir=directory
        )
        manifest_tmp = Path(name)
        with os.fdopen(fd, "w") as output:
            json.dump(manifest, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(manifest_tmp, directory / MANIFEST_NAME)
        manifest_tmp = None
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return table
    finally:
        for path in (temporary, manifest_tmp):
            if path is not None:
                path.unlink(missing_ok=True)
        os.close(lock_fd)


def cache_allocation_bytes(cache_rows: int, row_bytes: int, queue_depth: int) -> int:
    """Account for the imported cache's arena, row index and I/O staging."""
    if cache_rows < 0 or row_bytes <= 0 or not 1 <= queue_depth <= 4096:
        raise ValueError("Invalid NVMe cache allocation parameters")
    staging_span = (
        (row_bytes + DIO_ALIGNMENT * 2 - 1) // DIO_ALIGNMENT * DIO_ALIGNMENT
    )
    staging = queue_depth * staging_span
    if cache_rows == 0:
        return staging
    hash_capacity = 1 << (2 * cache_rows - 1).bit_length()
    arena = (
        (cache_rows * row_bytes + DIO_ALIGNMENT - 1)
        // DIO_ALIGNMENT
        * DIO_ALIGNMENT
    )
    return arena + cache_rows * 5 + hash_capacity * 8 + staging


def cache_rows_for_budget(
    budget_bytes: int, row_bytes: int, queue_depth: int, total_rows: int
) -> int:
    """Choose rows within a total cache budget, including the row index."""
    if total_rows <= 0 or total_rows >= 0xFFFFFFFF:
        raise ValueError("Invalid number of INT4 NVMe rows")
    if budget_bytes < cache_allocation_bytes(1, row_bytes, queue_depth):
        raise ValueError("NVMe cache budget is too small for staging and one row")
    lo, hi = 1, min(total_rows, budget_bytes // row_bytes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cache_allocation_bytes(mid, row_bytes, queue_depth) <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


class PleNvmeInt4Table:
    """Serve INT4 rows through the unchanged ple_nvme_ext.NvmeTable ABI."""

    is_nvme = True
    is_fp8 = False
    layout = INT4_LAYOUT

    def __init__(
        self,
        quant_dir: str,
        directory: str,
        total_rows: int,
        width: int,
        cache_gb: float = 8,
        queue_depth: int = 512,
    ) -> None:
        if not math.isfinite(cache_gb) or cache_gb <= 0:
            raise ValueError("NVMe cache GiB must be finite and positive")
        self.quant_dir = quant_dir
        self.width = width
        self._table = None
        self._lock_fd = None
        directory_path = Path(directory).resolve()
        spec = _source_spec(Path(quant_dir))
        if spec["rows"] != total_rows or spec["width"] != width:
            raise ValueError("INT4 NVMe sidecar does not match the model table")
        self.n_shards = spec["shards"]
        self._row_bytes = spec["row_bytes"]
        self._total_rows = total_rows
        self._cache_rows = cache_rows_for_budget(
            int(cache_gb * GIB), self._row_bytes, queue_depth, total_rows
        )
        try:
            from ple_nvme_ext import NvmeTable
        except ImportError as error:
            raise RuntimeError(
                "INT4 NVMe requires the CPU ple_nvme_ext extension; "
                "see docs/rdna2/PLE-NVME.md"
            ) from error
        try:
            self._lock_fd = _lock(directory_path)
            _validate_table(directory_path, spec)
            self._table = NvmeTable(
                str(directory_path / TABLE_NAME),
                total_rows,
                self._row_bytes,
                self._cache_rows,
                queue_depth,
            )
            stats = self._table.stats()
            if stats["align"] > DIO_ALIGNMENT:
                raise ValueError("INT4 NVMe supports direct-I/O alignment up to 4096")
            logger.info(
                "PLE INT4 NVMe: %d cached rows, %.2f GiB cache budget, "
                "%s, direct-I/O alignment %d (no whole-table prefault)",
                self._cache_rows,
                cache_gb,
                stats["backend"],
                stats["align"],
            )
            if stats["unlocked_bytes"]:
                logger.warning(
                    "PLE INT4 NVMe: mlock refused %.2f GiB; raise memlock "
                    "limits to prevent cache swapping",
                    stats["unlocked_bytes"] / GIB,
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._table = None
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def stats(self) -> dict:
        if self._table is None:
            raise RuntimeError("INT4 NVMe table is closed")
        return self._table.stats()

    def gather_rows_small(self, ids: np.ndarray, out: np.ndarray) -> None:
        if self._table is None:
            raise RuntimeError("INT4 NVMe table is closed")
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if ids.ndim != 1 or out.shape != (ids.size, self.width):
            raise ValueError("Invalid INT4 NVMe gather shape")
        if out.dtype != np.float32:
            raise ValueError("INT4 NVMe gather requires float32 output")
        if not ids.size:
            return
        if ids.min() < 0 or ids.max() >= self._total_rows:
            raise ValueError("INT4 NVMe row ID out of range")
        rows = np.empty((ids.size, self._row_bytes), dtype=np.uint8)
        self._table.gather(
            ids.ctypes.data, ids.size, rows.ctypes.data, self._row_bytes
        )
        packed = rows[:, : self.width // 2]
        scales = rows[:, self.width // 2 :].copy().view(np.float16)
        nib = np.empty((ids.size, self.width), dtype=np.float32)
        nib[:, 0::2] = packed & 0xF
        nib[:, 1::2] = packed >> 4
        out[:] = (nib - 8.0) * np.repeat(
            scales.astype(np.float32), 16, axis=1
        )

    def gather_into(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        if ids.device.type != "cpu" or out.device.type != "cpu":
            raise ValueError("INT4 NVMe gathers must execute in the CPU worker")
        if ids.ndim != 1 or tuple(out.shape) != (ids.numel(), self.width):
            raise ValueError("Invalid INT4 NVMe gather shape")
        rows = np.empty((ids.numel(), self.width), dtype=np.float32)
        self.gather_rows_small(ids.long().numpy(), rows)
        out.copy_(torch.from_numpy(rows).to(out.dtype))
