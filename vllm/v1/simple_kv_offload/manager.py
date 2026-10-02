# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload import debug as _offload_debug
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# gfx1030 fork: upstream lazy walk resumed from a cursor (see
# _prepare_lazy_store_specs). 1 restores it.
_LAZY_RESUME_CURSOR = os.getenv("VLLM_RDNA_OFFLOAD_LAZY_CURSOR", "0") == "1"
# gfx1030 fork: copy blocks the allocator evicts uncopied (see _on_gpu_evict).
_LAZY_RESCUE = os.getenv("VLLM_RDNA_OFFLOAD_LAZY_RESCUE", "1") == "1"


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
        disk_capacity_bytes: int = 0,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # When disk mode is active, the offload pool size is disk-based.
        offload_capacity = (
            disk_capacity_bytes if disk_capacity_bytes > 0 else cpu_capacity_bytes
        )
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.cp_world_size = dcp_world_size
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, offload_capacity
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        # Find the full attention kv group for prefix cache matching.
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)
        # FA group's own block_size; divides scheduler_block_size (the LCM)
        # but is NOT assumed to equal it.
        self.fa_block_size: int = (
            self.cpu_kv_cache_config.kv_cache_groups[
                self.fa_gidx
            ].kv_cache_spec.block_size
            * self.cp_world_size
        )
        assert self.block_size % self.fa_block_size == 0

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d offload blocks "
            "(%.2f GB, mode=%s, backend=%s)",
            self.num_cpu_blocks,
            offload_capacity / (1024**3),
            "lazy" if lazy_offload else "eager",
            "disk" if disk_capacity_bytes > 0 else "cpu",
        )
        if _offload_debug.ENABLED:
            _offload_debug.describe_groups(self.cpu_kv_cache_config)

        spec_config = vllm_config.speculative_config
        use_eagle = spec_config is not None and spec_config.use_eagle()
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length) tuples from find_longest_cache_hit,
        # kept pinned via touch() while awaiting update_state_after_alloc().
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int]
        ] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        self._lazy_steps = 0
        self._lazy_step_marks: dict[int, tuple] = {}
        self._parent_key: dict = {}
        self._chain_refresh_marks: dict = {}
        self._refresh_covered: set = set()
        self._root_of: dict = {}  # attention key -> key of its prompt's first block
        self._use_stamp: dict = {}  # prompt root key -> scheduler step of last use
        self._cpu_stamp: dict[int, int] = {}  # RAM block id -> last-use stamp
        # Eviction-time rescue (see _on_gpu_evict).
        self._rescue_gpu: list[int] = []
        self._rescue_cpu: list[int] = []
        self._rescue_keys: set = set()
        self._rescue_maturing: list[tuple[int, list[int]]] = []
        self._maturing_keys: set = set()
        self._chain: dict[int, tuple] = {}
        self._costore_groups: list[int] = []
        self._meta_steps = 0
        self._rescue_stats = [0, 0, 0]  # rescued, co-stored, lost (no CPU block)
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
            logger.info("SimpleCPUOffloadScheduler: lazy store window %d GPU blocks",
                        self._target_free)
        else:
            self._target_free = 0
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        self._abandoned_store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        self._in_flight_store_gpu_blocks: set[int] = set()
        self._abandoned_reqs_to_load: dict[str, LoadRequestState] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        # Every KVCacheTensor describes placement within the same backing allocation,
        # so its size is the total GPU KV cache size.
        gpu_total_bytes = gpu_config.kv_cache_tensors[0].size
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                layers=list(t.layers),
                layer_stride=t.layer_stride,
                block_stride=t.block_stride,
                offset=t.offset,
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return replace(
            gpu_config,
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        cp_world_size: int = 1,
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            block_size = spec.block_size * cp_world_size
            if isinstance(spec, MambaSpec):
                target += 2
            elif not getattr(spec, "prefix_cacheable", True):
                # gfx1030 fork: e.g. CircularBufferSpec (QSA ring, block_size
                # 4): one block per request for its lifetime, never cached.
                # Counting it as max_num_batched_tokens / 4 gave a window of
                # ~1000 blocks -- the whole free queue -- so lazy stored every
                # block as soon as it was freed (eager behaviour, no extra
                # capacity) and the CPU tier's LRU dropped older prefixes.
                target += 1
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        override = int(os.getenv("VLLM_RDNA_OFFLOAD_LAZY_TARGET", "0"))
        if override > 0:
            return override
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool
        if self._lazy_mode and not _LAZY_RESUME_CURSOR and _LAZY_RESCUE:
            evict = gpu_block_pool._maybe_evict_cached_block

            def evict_with_rescue(block: "KVCacheBlock") -> bool:
                self._on_gpu_evict(block)
                return evict(block)

            gpu_block_pool._maybe_evict_cached_block = evict_with_rescue  # type: ignore[method-assign]
            self._costore_groups = [
                g for g, grp in enumerate(self.cpu_kv_cache_config.kv_cache_groups)
                if grp.kv_cache_spec.block_size * self.cp_world_size == self.hash_block_size
                and getattr(grp.kv_cache_spec, "prefix_cacheable", True)
            ]

    def _on_gpu_evict(self, block: "KVCacheBlock") -> None:
        """gfx1030 fork: the GPU allocator is about to reuse a cached block.

        The lazy walk copies blocks shortly before they reach the eviction end,
        but a block can still be evicted uncopied (window used up by in-flight
        copies, or one allocation larger than the window, e.g. admitting a long
        reload). One lost block cuts every prefix through it: measured, a 178k
        prompt split GPU 0-15 / CPU 17+ with block 16 nowhere -> full recompute.
        Here the block gets a CPU block now and the worker copies it at the
        start of this step, before the block is zeroed for its new owner.

        Also copies the same-prefix blocks of the other groups still cached on
        the GPU (Mamba state snapshots age separately in the LRU): a CPU hit
        needs the state at its end position in RAM, not on the GPU.
        """
        bhash = block.block_hash
        if bhash is None or block.is_null:
            return
        cpu_pool = self.cpu_block_pool
        hit = cpu_pool.cached_block_hash_to_block.get_one_block(bhash)
        if hit is not None:
            return  # already in RAM (recency is set on use: see _note_use)
        for src, key in self._plan_copies(block, bhash, self._rescue_keys):
            if cpu_pool.get_num_free_blocks() == 0:
                self._rescue_stats[2] += 1
                continue
            cpu_blk = cpu_pool.get_new_blocks(1)[0]
            cpu_blk._block_hash = key  # type: ignore[assignment]
            self._rescue_gpu.append(src.block_id)
            self._rescue_cpu.append(cpu_blk.block_id)
            self._rescue_stats[0 if src is block else 1] += 1

    def _plan_copies(self, block: "KVCacheBlock", bhash, exclude: set) -> list:
        """gfx1030 fork: what to copy to CPU along with ``block``.

        A CPU hit on this hybrid model needs, from the point where the GPU hit
        ends, every attention block of the prefix plus the Mamba state at the
        hit's end -- all in RAM. vLLM frees a request's blocks tail first, so
        the head of a prefix stays on the GPU longest, yet without a Mamba
        state near the start it gives no GPU hit (measured: GPU kept blocks
        0-1, RAM had 2+, reload recomputed 178k tokens). So with a block also
        copy (a) the same-prefix blocks of the other cacheable groups still on
        the GPU and (b) its prefix ancestors still on the GPU (recorded per
        request in _record_chain), each with (a). Keys in ``exclude`` or already
        in RAM are skipped; chosen keys are added to ``exclude``.
        """
        out: list = []
        cached = self.cpu_block_pool.cached_block_hash_to_block
        gpu_pool = self._gpu_block_pool
        gpu_cached = gpu_pool.cached_block_hash_to_block

        def want(key) -> bool:
            return (key not in exclude and key not in self._maturing_keys
                    and cached.get_one_block(key) is None)

        def add_position(src, key) -> None:
            if want(key):
                out.append((src, key))
                exclude.add(key)
            raw, gk = get_block_hash(key), get_group_id(key)
            for g in self._costore_groups:
                if g == gk:
                    continue
                k2 = make_block_hash_with_group_id(raw, g)
                if not want(k2):
                    continue
                b2 = gpu_cached.get_one_block(k2)
                if b2 is not None and not b2.is_null:
                    out.append((b2, k2))
                    exclude.add(k2)

        add_position(block, bhash)
        # Follow the prefix from this position's attention block (also when the
        # walk met a Mamba block first: its attention block's ancestors are
        # needed just the same).
        head = block
        if get_group_id(bhash) != self.fa_gidx:
            k_fa = make_block_hash_with_group_id(get_block_hash(bhash), self.fa_gidx)
            head = gpu_cached.get_one_block(k_fa)
            bhash = k_fa
        entry = self._chain.get(head.block_id) if head is not None else None
        if entry is not None and entry[0] == bhash:
            prev, n = entry[1], 0
            while prev >= 0 and n < 4096:
                e = self._chain.get(prev)
                b = gpu_pool.blocks[prev]
                if e is None or b.block_hash != e[0] or b.is_null:
                    break  # ancestor evicted/reused: chain ends on the GPU
                if cached.get_one_block(e[0]) is not None:
                    break  # ancestor already in RAM (and its ancestors, normally)
                add_position(b, e[0])
                prev, n = e[1], n + 1
        return out

    def _record_chain(self, block_ids: tuple[list[int], ...]) -> None:
        """Remember each attention block's predecessor for _plan_copies."""
        if not (self._lazy_mode and not _LAZY_RESUME_CURSOR and _LAZY_RESCUE):
            return
        if self.fa_gidx >= len(block_ids):
            return
        pool = self._gpu_block_pool
        prev = -1
        prev_key = None
        if len(self._parent_key) > 400_000:
            self._parent_key.clear()
            self._chain_refresh_marks.clear()
        root = None
        if len(self._root_of) > 400_000:
            self._root_of.clear()
        for bid in block_ids[self.fa_gidx]:
            blk = pool.blocks[bid]
            if blk.is_null or blk.block_hash is None:
                break
            self._chain[bid] = (blk.block_hash, prev)
            self._parent_key[blk.block_hash] = prev_key
            root = root or blk.block_hash
            self._root_of[blk.block_hash] = root
            if _offload_debug.ENABLED:
                _offload_debug.note_root(blk.block_hash, root)
            prev, prev_key = bid, blk.block_hash
        if root is not None:
            self._use_stamp[root] = self._lazy_steps  # last use: finished now

    def _mature_rescues(self) -> None:
        """Make rescued CPU blocks findable two steps after their copy was
        issued: by then the copy (enqueued before that step's forward) has
        completed, even with one step of async scheduling overlap."""
        self._meta_steps += 1
        cpu_pool = self.cpu_block_pool
        while self._rescue_maturing and self._rescue_maturing[0][0] <= self._meta_steps - 2:
            _, cpu_ids = self._rescue_maturing.pop(0)
            blocks = [cpu_pool.blocks[b] for b in cpu_ids]
            for b in blocks:
                cpu_pool.cached_block_hash_to_block.insert(b.block_hash, b)
                self._maturing_keys.discard(b.block_hash)
            self._place_new_cpu_blocks(blocks)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""

        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]
        if request.block_hashes and self._lazy_mode and not _LAZY_RESUME_CURSOR and _LAZY_RESCUE:
            # gfx1030 fork: a request uses its prefix -> RAM copy to MRU (whole
            # prefix, tail to head). RAM recency follows last use; refreshing
            # on GPU eviction instead kept bumping a long prompt that leaves the
            # GPU a few blocks per step, past newer prefixes (measured: a 178k
            # prompt last used before A outlived A's tail).
            self._use_stamp[make_block_hash_with_group_id(request.block_hashes[0], self.fa_gidx)] = (
                self._lazy_steps)
            self._refresh_chain(
                make_block_hash_with_group_id(request.block_hashes[-1], self.fa_gidx),
                self._refresh_covered, force=True)

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False
        cpu_hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )
        if _offload_debug.ENABLED and len(request.block_hashes) >= 8:
            k0 = make_block_hash_with_group_id(request.block_hashes[0], self.fa_gidx)
            logger.info("offload-debug: lookup %s prompt %s: GPU hit %d tok, RAM hit %d of %d tok",
                        request.request_id[:24], bytes(k0)[:3].hex(), num_computed_tokens,
                        hit_length, max_hit_len)
            if 0 < hit_length < 0.95 * max_hit_len:
                _offload_debug.explain_miss(request, remaining_hashes, self.cpu_block_pool,
                                            self.cpu_kv_cache_config, self.hash_block_size,
                                            self._gpu_block_pool)

        if hit_length > 0:
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = (
                cpu_hit_blocks,
                hit_length,
            )
            return hit_length, True
        if _offload_debug.ENABLED and len(remaining_hashes) >= 8:
            _offload_debug.explain_miss(request, remaining_hashes, self.cpu_block_pool,
                                        self.cpu_kv_cache_config, self.hash_block_size,
                                        self._gpu_block_pool)
        return 0, False

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _ = pending

        # ``num_external_tokens`` is LCM-aligned (checked per-group below),
        # so this counts whole scheduler-aligned chunks of incoming tokens.
        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0
        num_cached_fa_blocks = sum(
            blk.block_hash is not None for blk in blocks.blocks[self.fa_gidx]
        )
        num_computed_tokens = num_cached_fa_blocks * self.fa_block_size

        # Build transfer pairs across all groups.
        total_computed_tokens = num_computed_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            g_block_size = (
                kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
            )
            assert num_external_tokens % g_block_size == 0, (
                f"num_external_tokens={num_external_tokens} not aligned to "
                f"group {g} block_size={g_block_size}"
            )
            n_take_g = num_external_tokens // g_block_size
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            # Number of blocks in the computed range for this group.
            g_block_size = (
                kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
            )
            n_computed_g = cdiv(total_computed_tokens, g_block_size)

            # Back-trace: ext blocks sit at the tail of the computed range.
            gpu_ext_start = n_computed_g - n_ext_g
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request, transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids)
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        # --- Eviction-time rescues (lazy) ---
        self._refresh_covered = set()
        self._mature_rescues()
        rescue_gpu, rescue_cpu = self._rescue_gpu, self._rescue_cpu
        if rescue_cpu:
            self._rescue_maturing.append((self._meta_steps, list(rescue_cpu)))
            self._maturing_keys.update(self._rescue_keys)
            self._rescue_gpu, self._rescue_cpu = [], []
        self._rescue_keys = set()

        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu
            )
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids

        if _offload_debug.ENABLED:
            cpu_blocks = self.cpu_block_pool.blocks
            _offload_debug.note_stored(cpu_blocks[b].block_hash for b in store_cpu + rescue_cpu)
        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs={
                event_idx: list(req_ids)
                for event_idx, req_ids in self._load_event_to_reqs.items()
            },
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            need_flush=bool(scheduler_output.preempted_req_ids),
            rescue_gpu_blocks=rescue_gpu,
            rescue_cpu_blocks=rescue_cpu,
        )
        return result

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Prepare store specs for the store event."""
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _place_new_cpu_blocks(self, blocks: list) -> None:
        """gfx1030 fork: release freshly copied RAM blocks into the RAM LRU
        relative to their prefix instead of at the MRU end.

        Each attention block goes just before (older than) its parent block if
        that is in RAM, so a prefix is evicted tail first; Mamba blocks go just
        before the attention block of their position. Blocks without a parent
        in RAM are appended (new prefix). Only a request using a prefix moves
        it as a whole (_refresh_chain from get_num_new_matched_tokens) --
        appending copies at MRU instead let a new tail block, written while an
        old 178k prompt was decoded again, drag that whole prompt past newer
        ones.
        """
        q = self.cpu_block_pool.free_block_queue
        cached = self.cpu_block_pool.cached_block_hash_to_block
        gpu_cached = self._gpu_block_pool.cached_block_hash_to_block
        fa = self.fa_gidx

        stamps = self._cpu_stamp
        now = self._lazy_steps

        def insert_before(b, nxt) -> None:
            prev = nxt.prev_free_block
            b.prev_free_block, b.next_free_block = prev, nxt
            prev.next_free_block = b
            nxt.prev_free_block = b
            q.num_free_blocks += 1

        def last_use(b) -> int:
            k = b.block_hash
            if get_group_id(k) != fa:
                k = make_block_hash_with_group_id(get_block_hash(k), fa)
            root = self._root_of.get(k)
            return self._use_stamp.get(root, now) if root is not None else now

        def release(b, anchor_key, is_root: bool) -> None:
            b.ref_cnt -= 1
            if b.ref_cnt > 0 or b.is_null:
                return
            anchor = cached.get_one_block(anchor_key) if anchor_key is not None else None
            if (anchor is not None and anchor is not b and anchor.ref_cnt == 0
                    and anchor.prev_free_block is not None):
                insert_before(b, anchor)
                stamps[b.block_id] = stamps.get(anchor.block_id, now)
            elif (not is_root and anchor_key is not None
                  and gpu_cached.get_one_block(anchor_key) is None):
                # Orphan: its parent (or, for a Mamba block, the attention
                # block of its position) is neither in RAM nor on the GPU, so
                # it can never be hit -- evict it first instead of letting it
                # outlive real prefixes.
                q.prepend_n([b])
                stamps[b.block_id] = -1
            else:
                # New prefix start (or anchor still on the GPU): rank by the
                # prompt's last use, not by copy time -- a prompt used before A
                # but copied after A must still be evicted before A.
                s = last_use(b)
                cur = q.fake_free_list_tail
                while (cur.prev_free_block is not q.fake_free_list_head
                       and stamps.get(cur.prev_free_block.block_id, -1) > s):
                    cur = cur.prev_free_block
                insert_before(b, cur)
                stamps[b.block_id] = s

        # Copy order is tail first; place head first so parents are in place.
        attn = [b for b in blocks if get_group_id(b.block_hash) == fa]
        other = [b for b in blocks if get_group_id(b.block_hash) != fa]
        for b in reversed(attn):
            parent = self._parent_key.get(b.block_hash, False)
            # parent None = recorded first block; False = chain unknown (keep as new)
            release(b, parent or None, is_root=not parent)
        for b in reversed(other):
            release(b, make_block_hash_with_group_id(get_block_hash(b.block_hash), fa), False)

    def _refresh_chain(self, key, covered: set | None = None, force: bool = False) -> None:
        """gfx1030 fork: move a prefix's RAM copy to MRU, tail first, head last.

        A RAM hit on this model needs the prefix unbroken from its first block,
        with the Mamba state blocks of its positions. If any block of it is
        older in the RAM tier's LRU than the rest, it is dropped first and the
        whole prefix becomes useless. Measured: the blocks at Mamba retention
        positions reached RAM early (co-copied when the walk met a Mamba block
        first), were never refreshed, and every 8th block of two 48k prompts
        vanished under pressure -> 0 % hits. So recency is maintained per
        prefix: walking parent keys (recorded in _record_chain, independent of
        the GPU) from ``key`` to the root, every position's blocks of all
        cacheable groups are moved to MRU in tail-to-head order, so eviction
        takes tails first. Used whenever blocks are copied, or a block already
        in RAM leaves the GPU (it was in use: refresh its prefix).
        """
        fa = self.fa_gidx
        if get_group_id(key) != fa:
            key = make_block_hash_with_group_id(get_block_hash(key), fa)
        if covered is not None and key in covered:
            return
        if not force:
            last = self._chain_refresh_marks.get(key)
            if last is not None and self._lazy_steps - last < 64:
                return
        self._chain_refresh_marks[key] = self._lazy_steps
        positions = []
        k, n = key, 0
        while k is not None and n < 8192:
            if covered is not None:
                if k in covered:
                    break
                covered.add(k)
            positions.append(k)
            k = self._parent_key.get(k)
            n += 1
        q = self.cpu_block_pool.free_block_queue
        cached = self.cpu_block_pool.cached_block_hash_to_block
        groups = self._costore_groups or [fa]
        for k in positions:
            raw = get_block_hash(k)
            for g in groups:
                b = cached.get_one_block(k if g == fa else make_block_hash_with_group_id(raw, g))
                if b is not None and b.ref_cnt == 0 and not b.is_null:
                    q.remove(b)
                    q.append(b)
                    self._cpu_stamp[b.block_id] = self._lazy_steps

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[str]]:
        """Single-pass cursor walk: offload cached GPU blocks near eviction.

        Walks the GPU free queue from the cursor, counting blocks that are
        free-or-offloaded (safe for the allocator to evict). Stops when
        target_free blocks are covered or CPU capacity is reached.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return [], [], []

        free_queue = gpu_pool.free_block_queue
        cpu_pool = self.cpu_block_pool
        num_cpu_free = cpu_pool.get_num_free_blocks()

        # Validate cursor: stale if block was removed from free queue.
        cursor_reset = self._cursor is not None and self._cursor.ref_cnt > 0
        if cursor_reset:
            self._cursor = None
        # gfx1030 fork: measure the window from the head (next to be evicted)
        # every step. Resuming from the cursor extended the window by
        # target_free blocks per step, so over a long chunked prefill the walk
        # ran deep into the queue and stored nearly every block (eager
        # behaviour); the CPU tier's LRU then dropped the oldest prefixes.
        # Rewalking costs <= target_free hash lookups per step.
        self._lazy_steps += 1
        window = self._target_free
        if not _LAZY_RESUME_CURSOR:
            self._cursor = None
            # Blocks whose store is still in flight are out of the free queue
            # (touched) and return to its head when done, so they count toward
            # the window. Otherwise each step walked target_free blocks past
            # them, and while completions lagged a few steps (e.g. a CPU reload
            # followed by decode) hundreds of blocks were stored at once,
            # draining the CPU tier and evicting its older prefixes.
            window -= sum(
                len(t.gpu_block_ids) for t in self._store_event_to_blocks.values()
            )
            if window <= 0:
                return [], [], []

        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []
        last_visited = self._cursor
        plan_chain = not _LAZY_RESUME_CURSOR and _LAZY_RESCUE
        walk_keys: set = set(self._rescue_keys)

        for covered, node in enumerate(free_queue.iter_blocks_after(self._cursor)):
            if covered >= window or len(gpu_ids) >= num_cpu_free:
                break

            last_visited = node
            bhash = node.block_hash

            cpu_blk = (
                cpu_pool.cached_block_hash_to_block.get_one_block(bhash)
                if bhash is not None and not node.is_null
                else None
            )
            stored = (
                bhash is not None
                and not node.is_null
                and cpu_blk is None
                and bhash not in self._maturing_keys
                and bhash not in walk_keys
            )
            if stored:
                if plan_chain:
                    for src, key in self._plan_copies(node, bhash, walk_keys):
                        if len(gpu_ids) >= num_cpu_free:
                            break
                        gpu_ids.append(src.block_id)
                        block_hashes.append(key)
                else:
                    gpu_ids.append(node.block_id)
                    block_hashes.append(bhash)
            if _offload_debug.ENABLED:
                _offload_debug.note_walk(node, stored)

        self._cursor = last_visited
        if _offload_debug.ENABLED:
            _offload_debug.maybe_report(self._target_free, num_cpu_free, cursor_reset,
                                        self._rescue_stats, self.cpu_block_pool, self.fa_gidx)

        # Batch-allocate CPU blocks and stamp hashes.
        if gpu_ids:
            cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            for cpu_blk, bhash in zip(cpu_blocks, block_hashes):  # type: ignore[assignment]
                cpu_blk._block_hash = bhash  # type: ignore[assignment]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, []

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. This means blocks from the current step are NOT stored until the
        next step. If a request finishes in the same step as its last full block,
        that block may be missed. (TODO: flush on finish.)

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids) for the store event.
        """

        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], []
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)
        # Dedup against blocks already scheduled.
        in_flight = self._in_flight_store_gpu_blocks

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            # Accumulate new block IDs.
            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            advanced_per_group: list[int] = [0] * num_groups
            out_of_space = False
            # Confirmed tokens: KV data written and visible to all streams.
            req = state.request
            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders
            # Cap to blocks with confirmed KV data.
            aligned_tokens = confirmed_tokens // self.block_size * self.block_size

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                already_stored_g = state.num_stored_blocks[g]
                group_gpu_ids = block_ids_by_group[g]

                g_block_size = (
                    kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
                )
                ready_blocks_g = aligned_tokens // g_block_size
                scannable = group_gpu_ids[already_stored_g:ready_blocks_g]

                for gpu_block_id in scannable:
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[g] += 1
                        continue

                    bhash_with_group = gpu_block.block_hash
                    if bhash_with_group is None:
                        # Masked-out SWA position the coordinator chose not to
                        # hash; it can never serve a prefix-cache hit, so skip.
                        advanced_per_group[g] += 1
                        continue

                    # Skip if already scheduled for store or already cached in CPU.
                    if (
                        gpu_block_id in in_flight
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        advanced_per_group[g] += 1
                        continue

                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
                for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes_to_store):
                    cpu_blk._block_hash = bhash  # type: ignore[assignment]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                in_flight.update(gpu_block_ids)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # --- Load completions ---
        for req_id in list(connector_output.finished_recving or []):
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx, None)
        if transfer is None:
            transfer = self._abandoned_store_event_to_blocks.pop(event_idx, None)
            if transfer is None:
                return  # guard stale events from before a reset() call
            self._release_transfer_refs(transfer)
            return

        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)

        self._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(
        self, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        Block hashes were stamped on CPU blocks at allocation time (in
        ``_prepare_*_store_specs``).  Here we just register them in the
        cache map so they become discoverable by the load path.
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for cpu_block in cpu_blocks:
            bhash = cpu_block.block_hash
            assert bhash is not None
            self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        if self._lazy_mode and not _LAZY_RESUME_CURSOR and _LAZY_RESCUE:
            self._place_new_cpu_blocks(cpu_blocks)
        else:
            self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        if self._lazy_mode and not _LAZY_RESUME_CURSOR:
            # gfx1030 fork: the store touched these GPU blocks (out of the free
            # queue); free_blocks() would append them at the MRU tail, keeping
            # every offloaded block on the GPU for another full LRU cycle --
            # the tiers stay inclusive and lazy gains no capacity. They were
            # next to be evicted and are now safe on CPU: return them to the
            # head.
            pool = self._gpu_block_pool
            to_head = []
            for bid in gpu_block_ids:
                blk = pool.blocks[bid]
                blk.ref_cnt -= 1
                if blk.ref_cnt == 0 and not blk.is_null:
                    to_head.append(blk)
            pool.free_block_queue.prepend_n(to_head)
            return
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
        )

    def _release_transfer_refs(self, transfer: TransferMeta) -> None:
        """Release transfer refs without making copied data cacheable."""
        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in transfer.cpu_block_ids]
        for cpu_block in cpu_blocks:
            cpu_block.reset_hash()
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in transfer.gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers."""
        return bool(
            self._store_event_to_blocks or self._abandoned_store_event_to_blocks
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._record_chain(block_ids)
        return self.request_finished(request, block_ids=[])

    def _free_pending_cpu_hit(self, pending: tuple) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens()."""
        cpu_hit_blocks, _ = pending
        blocks_to_free = [
            blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self.cpu_block_pool.free_blocks(blocks_to_free)

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            state = self._abandoned_reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        if state.transfer_meta is not None:
            # Free CPU touch refs
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # gfx1030 fork: those were appended at MRU -- used now.
            for bid in state.transfer_meta.cpu_block_ids:
                self._cpu_stamp[bid] = self._lazy_steps
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()

    def reset(self) -> bool:
        """Abandon pending transfers and reset the CPU cache when safe.

        Worker-side DMA may still be using blocks after reset is requested.
        Keep those block refs pinned until the existing completion path reports
        the transfer finished, then release refs without caching abandoned
        store results.
        """

        self._abandoned_store_event_to_blocks.update(self._store_event_to_blocks)
        self._store_event_to_blocks.clear()
        self._in_flight_store_gpu_blocks.clear()

        # Loads that have not been sent to the worker cannot have running DMA.
        # In-flight loads stay pinned and are cleaned up on completion.
        for req_id in list(self._reqs_to_load):
            state = self._reqs_to_load.pop(req_id)
            if state.load_event is None:
                self._reqs_to_load[req_id] = state
                self._cleanup_load_request(req_id)
            else:
                self._abandoned_reqs_to_load[req_id] = state

        self._reqs_to_store.clear()
        self._store_event_to_reqs.clear()
        self._store_event_pending_counts = {
            event_idx: count
            for event_idx, count in self._store_event_pending_counts.items()
            if event_idx in self._abandoned_store_event_to_blocks
        }
        self._cursor = None
        # NOTE: _load_event_counter / _store_event_counter are not
        # reset as they are monotonic and must stay ahead of the workers
        # high-water marks to avoid event index collisions

        if self._abandoned_store_event_to_blocks or self._abandoned_reqs_to_load:
            return False

        return self.cpu_block_pool.reset_prefix_cache()
