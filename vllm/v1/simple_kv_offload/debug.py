# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for the simple CPU KV offload's scheduler side (gfx1030 fork),
VLLM_RDNA_OFFLOAD_DEBUG=1.

Per KV-cache group: how many blocks the lazy cursor walk visited / found hashed /
stored, and, for a long request that gets no CPU hit, which group's blocks are
missing from the CPU tier (the hybrid model needs every group to hit).
"""

import os
import time
from collections import Counter

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_RDNA_OFFLOAD_DEBUG", "0") == "1"

_walk = Counter()
_last = 0.0
_stored_keys: dict = {}  # key -> times copied to CPU (bounded)
_root_of: dict = {}  # attention key -> key of its prompt's first block


def note_root(key, root) -> None:
    _root_of[key] = root
    if len(_root_of) > 400000:
        _root_of.clear()


def note_stored(keys) -> None:
    for k in keys:
        _stored_keys[k] = _stored_keys.get(k, 0) + 1
    if len(_stored_keys) > 200000:
        _stored_keys.clear()


def _gid(bhash) -> int:
    return int.from_bytes(bytes(bhash)[-4:], "big")


def describe_groups(kv_cache_config) -> None:
    for g, grp in enumerate(kv_cache_config.kv_cache_groups):
        spec = grp.kv_cache_spec
        logger.info("offload-debug: group %d: %s block_size=%d layers=%d", g,
                    type(spec).__name__, spec.block_size, len(grp.layer_names))


def note_walk(node, stored: bool) -> None:
    bhash = node.block_hash
    if bhash is None or node.is_null:
        _walk["visited/unhashed"] += 1
        return
    g = _gid(bhash)
    _walk[f"g{g}/hashed"] += 1
    if stored:
        _walk[f"g{g}/stored"] += 1


def maybe_report(target_free: int, num_cpu_free: int, cursor_reset: bool, rescue=None,
                  cpu_pool=None, fa_gidx: int = 0) -> None:
    global _last
    if cursor_reset:
        _walk["cursor_resets"] += 1
    now = time.monotonic()
    if now - _last < 30.0 or not _walk:
        return
    _last = now
    logger.info("offload-debug: lazy walk since last report (target_free=%d, cpu_free=%d): %s; "
                "eviction rescues so far (rescued, co-stored, lost): %s",
                target_free, num_cpu_free, dict(sorted(_walk.items())), rescue)
    _walk.clear()
    if cpu_pool is not None:
        from vllm.v1.core.kv_cache_utils import get_block_hash, make_block_hash_with_group_id
        per_root: Counter = Counter()
        pinned: Counter = Counter()
        for key, val in list(cpu_pool.cached_block_hash_to_block._cache.items()):
            k0 = key if _gid(key) == fa_gidx else make_block_hash_with_group_id(get_block_hash(key), fa_gidx)
            r = _root_of.get(k0)
            name = bytes(r)[:3].hex() if r is not None else "?"
            per_root[name] += 1
            blocks = val.values() if isinstance(val, dict) else [val]
            if any(b.ref_cnt > 0 for b in blocks):
                pinned[name] += 1
        logger.info("offload-debug: RAM tier by prompt (first-block id: blocks, all groups): %s; "
                    "pinned (ref>0): %s; free queue %d",
                    dict(per_root.most_common(30)), dict(pinned), cpu_pool.get_num_free_blocks())


def explain_miss(request, remaining_hashes, cpu_pool, kv_cache_config, hash_block_size, gpu_pool=None) -> None:
    """Long request, no CPU hit: per group, which of the first hashes are on CPU."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    n = min(len(remaining_hashes), 240)
    parts = []
    for g, grp in enumerate(kv_cache_config.kv_cache_groups):
        if grp.kv_cache_spec.block_size != hash_block_size:
            parts.append(f"g{g}:bs{grp.kv_cache_spec.block_size}(skipped)")
            continue
        def mark(h) -> str:
            k = make_block_hash_with_group_id(h, g)
            if cpu_pool.cached_block_hash_to_block.get_one_block(k) is not None:
                return "1"
            on_gpu = gpu_pool is not None and gpu_pool.cached_block_hash_to_block.get_one_block(k) is not None
            if k in _stored_keys:
                return "E" if not on_gpu else "e"  # copied earlier, gone from CPU (still on GPU: e)
            return "g" if on_gpu else "0"  # never copied: on GPU / nowhere
        present = "".join(mark(h) for h in remaining_hashes[:n])
        parts.append(f"g{g}:{present}")
    logger.info("offload-debug: CPU miss for %s (%d hashes): %s", request.request_id,
                len(remaining_hashes), " ".join(parts))
