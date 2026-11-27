"""Small HTTP helper with retries, pacing and a disk cache."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests


class Http:
    def __init__(self, cache_dir: str | Path, user_agent: str, min_interval: float = 0.15):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self._last = 0.0

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def _pace(self) -> None:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def json(self, method: str, url: str, body: dict | None = None, use_cache: bool = True,
             retries: int = 4) -> dict:
        key = method + url + json.dumps(body or {}, sort_keys=True)
        cp = self._cache_path(key)
        if use_cache and cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))
        last_err = None
        for attempt in range(retries):
            self._pace()
            try:
                resp = self.session.request(method, url, json=body, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    cp.write_text(json.dumps(data), encoding="utf-8")
                    return data
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}"
                    time.sleep(2 ** attempt)
                    continue
                raise HttpError(resp.status_code, resp.text[:500])
            except requests.RequestException as exc:  # network hiccup
                last_err = str(exc)
                time.sleep(2 ** attempt)
        raise HttpError(0, f"gave up after {retries} tries: {last_err}")


class HttpError(RuntimeError):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text}")
        self.status = status
