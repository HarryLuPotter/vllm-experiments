# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from typing import NamedTuple

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

# Allow for the time between a tool result reaching the API server and its
# request reaching the engine after admission and queueing.
QUEUE_DELAY_ALLOWANCE_SECONDS = 30.0


def reuse_deadline(now: float, predicted_round_trip_seconds: float) -> float:
    return now + predicted_round_trip_seconds + QUEUE_DELAY_ALLOWANCE_SECONDS


class FreeBlockEntry(NamedTuple):
    """Heap key uses LRU sequence, deadline, or negated deadline by heap."""

    priority: float | int
    sequence: int
    block_id: int
    generation: int


class FreeBlockSelectionIndex:
    
    """
    索引系统内 free blocks 的数据结构
    按照缓存状态、预计复用时间和 LRU 顺序指导选择待分配 block
    """

    def __init__(
        self, blocks: list[KVCacheBlock], free_block_queue: FreeKVCacheBlockQueue
    ) -> None:
        self.blocks = blocks
        self.free_block_queue = free_block_queue
        self._uncached_free_heap: list[FreeBlockEntry] = []
        self._deadline_min_heap: list[FreeBlockEntry] = []
        self._future_max_heap: list[FreeBlockEntry] = []
        self._expired_lru_heap: list[FreeBlockEntry] = []
        self._free_sequence = 0
        self._sequence_by_block = [0] * len(blocks)
        self._generation_by_block = [0] * len(blocks)
        self.rebuild()

    @property
    def num_entries(self) -> int:
        return (
            len(self._uncached_free_heap)
            + len(self._deadline_min_heap)
            + len(self._future_max_heap)
            + len(self._expired_lru_heap)
        )

    def _is_live(self, entry: FreeBlockEntry) -> bool:
        block = self.blocks[entry.block_id]
        return (
            self._generation_by_block[entry.block_id] == entry.generation
            and block.ref_cnt == 0
            and not block.is_null
            and block.prev_free_block is not None
            and block.next_free_block is not None
        )

    def on_free(self, block: KVCacheBlock) -> None:
        assert block.ref_cnt == 0 and not block.is_null
        block_id = block.block_id
        self._free_sequence += 1
        sequence = self._free_sequence
        self._sequence_by_block[block_id] = sequence
        self.invalidate(block)
        generation = self._generation_by_block[block_id]
        if block.block_hash is None:
            heapq.heappush(
                self._uncached_free_heap,
                FreeBlockEntry(sequence, sequence, block_id, generation),
            )
            return

        deadline = block.predicted_reuse_deadline
        if deadline is None:
            heapq.heappush(
                self._expired_lru_heap,
                FreeBlockEntry(sequence, sequence, block_id, generation),
            )
            return

        heapq.heappush(
            self._deadline_min_heap,
            FreeBlockEntry(deadline, sequence, block_id, generation),
        )
        heapq.heappush(
            self._future_max_heap,
            FreeBlockEntry(-deadline, sequence, block_id, generation),
        )

    def on_hash_removed(self, block: KVCacheBlock) -> None:
        assert block.ref_cnt == 0 and not block.is_null and block.block_hash is None
        self.invalidate(block)
        block_id = block.block_id
        sequence = self._sequence_by_block[block_id]
        heapq.heappush(
            self._uncached_free_heap,
            FreeBlockEntry(
                sequence, sequence, block_id, self._generation_by_block[block_id]
            ),
        )

    def invalidate(self, block: KVCacheBlock) -> None:
        self._generation_by_block[block.block_id] += 1

    def rebuild(self) -> None:
        self._uncached_free_heap.clear()
        self._deadline_min_heap.clear()
        self._future_max_heap.clear()
        self._expired_lru_heap.clear()
        self._free_sequence = 0
        for block in self.free_block_queue.get_all_free_blocks():
            if not block.is_null:
                self.on_free(block)

    def compact_if_needed(self) -> None:
        if self.num_entries > self.free_block_queue.num_free_blocks * 6:
            self.rebuild()

    def _promote_expired_blocks(self, now: float) -> None:
        # remaining_time = deadline - now, so deadline order stays unchanged;
        # only the one-way transition into the expired set needs refreshing.
        while self._deadline_min_heap:
            entry = self._deadline_min_heap[0]
            block = self.blocks[entry.block_id]
            if (
                not self._is_live(entry)
                or block.block_hash is None
                or block.predicted_reuse_deadline != entry.priority
            ):
                heapq.heappop(self._deadline_min_heap)
                continue
            if entry.priority > now:
                return
            heapq.heappop(self._deadline_min_heap)
            heapq.heappush(
                self._expired_lru_heap,
                FreeBlockEntry(
                    entry.sequence, entry.sequence, entry.block_id, entry.generation
                ),
            )

    def _pop_uncached_block(self) -> KVCacheBlock | None:
        while self._uncached_free_heap:
            entry = heapq.heappop(self._uncached_free_heap)
            block = self.blocks[entry.block_id]
            if self._is_live(entry) and block.block_hash is None:
                return block
        return None

    def _pop_expired_block(self, now: float) -> KVCacheBlock | None:
        while self._expired_lru_heap:
            entry = heapq.heappop(self._expired_lru_heap)
            block = self.blocks[entry.block_id]
            deadline = block.predicted_reuse_deadline
            if (
                self._is_live(entry)
                and block.block_hash is not None
                and (deadline is None or deadline <= now)
            ):
                return block
        return None

    def _pop_future_block(self, now: float) -> KVCacheBlock | None:
        while self._future_max_heap:
            entry = heapq.heappop(self._future_max_heap)
            block = self.blocks[entry.block_id]
            deadline = block.predicted_reuse_deadline
            if (
                self._is_live(entry)
                and block.block_hash is not None
                and deadline is not None
                and deadline > now
                and deadline == -entry.priority
            ):
                return block
        return None

    def pop_for_allocation(self, now: float) -> KVCacheBlock:
        self._promote_expired_blocks(now)
        block = self._pop_uncached_block()
        if block is None:
            block = self._pop_expired_block(now)
        if block is None:
            block = self._pop_future_block(now)
        if block is None:
            raise RuntimeError("Free-block selection index is out of sync")

        self.free_block_queue.remove(block)
        self.invalidate(block)
        return block
