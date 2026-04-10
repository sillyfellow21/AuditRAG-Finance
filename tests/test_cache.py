from __future__ import annotations

import time

from core.cache import TTLCache


def test_cache_hit_and_expiration() -> None:
    cache: TTLCache[str, str] = TTLCache(max_entries=4, ttl_seconds=1)
    cache.set("key", "value")

    assert cache.get("key") == "value"

    time.sleep(1.05)
    assert cache.get("key") is None

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] >= 1


def test_cache_eviction_on_max_entries() -> None:
    cache: TTLCache[str, int] = TTLCache(max_entries=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
