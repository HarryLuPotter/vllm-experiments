# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

pytestmark = pytest.mark.cpu_test


def make_pool(num_blocks: int = 4) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        hash_block_size=16,
    )


def cache_block(pool: BlockPool, block: KVCacheBlock) -> None:
    block_hash = make_block_hash_with_group_id(
        BlockHash(block.block_id.to_bytes(32, "big")), 0
    )
    block.block_hash = block_hash
    pool.cached_block_hash_to_block.insert(block_hash, block)


def test_uncached_block_is_allocated_before_cached_block(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool(3)
    cached = pool.get_new_blocks(1)[0]
    cache_block(pool, cached)
    pool.free_blocks([cached], predicted_reuse_deadline=200.0)

    allocated = pool.get_new_blocks(1)[0]
    assert allocated.block_id != cached.block_id


def test_future_block_with_latest_deadline_is_evicted_first(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool()
    early, latest, middle = pool.get_new_blocks(3)
    for block in (early, latest, middle):
        cache_block(pool, block)
    pool.free_blocks([early], predicted_reuse_deadline=110.0)
    pool.free_blocks([latest], predicted_reuse_deadline=130.0)
    pool.free_blocks([middle], predicted_reuse_deadline=120.0)

    assert pool.get_new_blocks(1)[0].block_id == latest.block_id
    assert pool.get_new_blocks(1)[0].block_id == middle.block_id
    assert pool.get_new_blocks(1)[0].block_id == early.block_id


def test_expired_and_missing_predictions_precede_future(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool()
    expired, missing, future = pool.get_new_blocks(3)
    for block in (expired, missing, future):
        cache_block(pool, block)
    pool.free_blocks([expired], predicted_reuse_deadline=90.0)
    pool.free_blocks([missing])
    pool.free_blocks([future], predicted_reuse_deadline=200.0)

    assert pool.get_new_blocks(1)[0].block_id == expired.block_id
    assert pool.get_new_blocks(1)[0].block_id == missing.block_id
    assert pool.get_new_blocks(1)[0].block_id == future.block_id


def test_deadline_is_promoted_after_time_advances(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "vllm.v1.core.block_pool.time.monotonic", lambda: now[0]
    )
    pool = make_pool(3)
    soon, later = pool.get_new_blocks(2)
    for block in (soon, later):
        cache_block(pool, block)
    pool.free_blocks([soon], predicted_reuse_deadline=105.0)
    pool.free_blocks([later], predicted_reuse_deadline=110.0)

    now[0] = 106.0
    assert pool.get_new_blocks(1)[0].block_id == soon.block_id


def test_equal_deadlines_preserve_lru_and_suffix_order(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool()
    prefix, middle, suffix = pool.get_new_blocks(3)
    for block in (prefix, middle, suffix):
        cache_block(pool, block)

    pool.free_blocks(
        reversed([prefix, middle, suffix]), predicted_reuse_deadline=200.0
    )

    assert pool.get_new_blocks(1)[0].block_id == suffix.block_id
    assert pool.get_new_blocks(1)[0].block_id == middle.block_id
    assert pool.get_new_blocks(1)[0].block_id == prefix.block_id


def test_shared_block_uses_deadline_from_last_free(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool(2)
    block = pool.get_new_blocks(1)[0]
    cache_block(pool, block)
    pool.touch([block])

    pool.free_blocks([block], predicted_reuse_deadline=110.0)
    assert block.ref_cnt == 1
    assert block.predicted_reuse_deadline is None

    pool.free_blocks([block], predicted_reuse_deadline=120.0)
    assert block.ref_cnt == 0
    assert block.predicted_reuse_deadline == 120.0


def test_touch_invalidates_old_heap_entry(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool(3)
    touched, other = pool.get_new_blocks(2)
    for block in (touched, other):
        cache_block(pool, block)
    pool.free_blocks([touched], predicted_reuse_deadline=300.0)
    pool.free_blocks([other], predicted_reuse_deadline=200.0)

    pool.touch([touched])
    assert touched.predicted_reuse_deadline is None
    assert pool.get_new_blocks(1)[0].block_id == other.block_id


def test_explicit_hash_eviction_reindexes_block_as_uncached(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool(3)
    evicted, cached = pool.get_new_blocks(2)
    for block in (evicted, cached):
        cache_block(pool, block)
    pool.free_blocks([evicted], predicted_reuse_deadline=200.0)
    pool.free_blocks([cached], predicted_reuse_deadline=300.0)

    pool.evict_blocks({evicted.block_id})

    assert evicted.block_hash is None
    assert evicted.predicted_reuse_deadline is None
    assert pool.get_new_blocks(1)[0].block_id == evicted.block_id


def test_prefix_reset_rebuilds_free_block_index(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool(3)
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        cache_block(pool, block)
        pool.free_blocks([block], predicted_reuse_deadline=200.0)

    assert pool.reset_prefix_cache()
    assert all(block.block_hash is None for block in blocks)
    assert all(block.predicted_reuse_deadline is None for block in blocks)
    assert pool.get_new_blocks(2) == blocks


def test_lazy_heap_entries_are_compacted(monkeypatch):
    monkeypatch.setattr("vllm.v1.core.block_pool.time.monotonic", lambda: 100.0)
    pool = make_pool(3)
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        cache_block(pool, block)
        pool.free_blocks([block], predicted_reuse_deadline=200.0)

    for _ in range(20):
        pool.touch([blocks[0]])
        pool.free_blocks([blocks[0]], predicted_reuse_deadline=200.0)

    num_heap_entries = (
        len(pool._uncached_free_heap)
        + len(pool._deadline_min_heap)
        + len(pool._future_max_heap)
        + len(pool._expired_lru_heap)
    )
    assert num_heap_entries <= pool.get_num_free_blocks() * 6


@pytest.mark.parametrize("now", [0.0, 100.0, 1000.0])
def test_heap_matches_full_remaining_time_scan(monkeypatch, now: float):
    monkeypatch.setattr(
        "vllm.v1.core.block_pool.time.monotonic", lambda: now
    )
    rng = random.Random(0)

    for _ in range(20):
        pool = make_pool(9)
        blocks = pool.get_new_blocks(8)
        for block in blocks:
            cache_block(pool, block)

        rng.shuffle(blocks)
        candidates: list[tuple[int, KVCacheBlock, float | None]] = []
        deadline_choices = [None, now - 5.0, now, now + 1.0, now + 10.0]
        for sequence, block in enumerate(blocks):
            deadline = rng.choice(deadline_choices)
            pool.free_blocks([block], predicted_reuse_deadline=deadline)
            candidates.append((sequence, block, deadline))

        while candidates:
            expired = [
                candidate
                for candidate in candidates
                if candidate[2] is None or candidate[2] <= now
            ]
            if expired:
                expected = min(expired, key=lambda candidate: candidate[0])
            else:
                future = [
                    (sequence, block, deadline)
                    for sequence, block, deadline in candidates
                    if deadline is not None
                ]
                expected = max(
                    future,
                    key=lambda candidate: (
                        candidate[2] - now,
                        -candidate[0],
                    ),
                )

            selected = pool.get_new_blocks(1)[0]
            assert selected.block_id == expected[1].block_id
            candidates.remove(expected)
