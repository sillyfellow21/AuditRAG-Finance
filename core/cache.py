from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from time import monotonic
from typing import Dict, Generic, Hashable, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(1, ttl_seconds)
        self._store: "OrderedDict[K, Tuple[float, V]]" = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> Optional[V]:
        now = monotonic()
        with self._lock:
            payload = self._store.get(key)
            if payload is None:
                self._misses += 1
                return None

            expires_at, value = payload
            if expires_at <= now:
                self._store.pop(key, None)
                self._misses += 1
                return None

            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        expires_at = monotonic() + self.ttl_seconds
        with self._lock:
            if key in self._store:
                self._store.pop(key, None)
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            self._evict_if_needed()

    def delete(self, key: K) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "size": len(self._store),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self._evictions += 1
