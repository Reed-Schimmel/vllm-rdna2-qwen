# SPDX-License-Identifier: Apache-2.0
"""Diagnostic timing for the simple CPU KV offload (gfx1030 fork), VLLM_RDNA_OFFLOAD_TIMING=1.

Records how long each batch-copy call of the offload's background thread takes, and logs worker steps whose
host-side wall time exceeds VLLM_RDNA_OFFLOAD_TIMING_SLOW_MS (default 500), with whether a copy was in flight.
A slow step while a copy is running means the copy thread is holding up the main thread (HIP lock contention),
which desynchronises the TP ranks.
"""

import os
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_RDNA_OFFLOAD_TIMING", "0") == "1"
SLOW_S = float(os.getenv("VLLM_RDNA_OFFLOAD_TIMING_SLOW_MS", "500")) / 1e3

_lock = threading.Lock()
_copy_since = 0.0          # start time of the copy in flight, 0 = idle
_copy_last = (0.0, 0.0, 0, False)  # (start, duration, descriptors, is_store) of the last finished copy


def copy_begin() -> float:
    global _copy_since
    t = time.monotonic()
    with _lock:
        _copy_since = t
    return t


def copy_end(t0: float, descriptors: int, is_store: bool) -> None:
    global _copy_since, _copy_last
    dt = time.monotonic() - t0
    with _lock:
        _copy_since = 0.0
        _copy_last = (t0, dt, descriptors, is_store)
    if dt > SLOW_S / 5:
        logger.warning("offload-timing: %s copy of %d descriptors took %.0f ms",
                       "store" if is_store else "load", descriptors, dt * 1e3)


def wrap_execute_model(fn):
    def wrapper(self, scheduler_output, *a, **kw):
        t0 = time.monotonic()
        try:
            return fn(self, scheduler_output, *a, **kw)
        finally:
            dt = time.monotonic() - t0
            if dt > SLOW_S:
                with _lock:
                    busy = _copy_since
                    last = _copy_last
                overlap = (busy and busy < t0 + dt) or (last[0] and last[0] < t0 + dt and last[0] + last[1] > t0)
                logger.warning(
                    "offload-timing: rank step took %.0f ms (%d tokens); copy in flight during step: %s; "
                    "last copy: %s %d desc %.0f ms", dt * 1e3,
                    getattr(scheduler_output, "total_num_scheduled_tokens", -1), bool(overlap),
                    "store" if last[3] else "load", last[2], last[1] * 1e3)
    return wrapper
