"""Round-robin API key rotation for the x.ai (Grok) backend.

Multiple free/rate-limited keys are supplied via the XAI_API_KEYS env var
(comma-separated, loaded from project .env — never committed to git). Each
call advances to the next key; on an auth/rate-limit/server error the caller
is retried against the remaining keys before giving up. This lets the RAG
generator survive a single key's rate limit without failing the request.
"""
import itertools
import os
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv():
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        # minimal manual fallback if python-dotenv isn't installed
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


class KeyRotator:
    """Thread-safe round-robin rotation over a list of API keys."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("KeyRotator requires at least one API key")
        self._keys = keys
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(range(len(keys)))

    def __len__(self):
        return len(self._keys)

    def next_key(self) -> str:
        with self._lock:
            idx = next(self._cycle)
            return self._keys[idx]

    def call_with_rotation(self, fn, max_attempts: int | None = None):
        """Call fn(api_key) -> result, retrying with the next key on failure.

        Tries at most `max_attempts` distinct keys (default: all of them)
        before re-raising the last exception.
        """
        attempts = max_attempts or len(self._keys)
        last_exc = None
        for _ in range(attempts):
            key = self.next_key()
            try:
                return fn(key)
            except Exception as e:  # noqa: BLE001 - deliberately broad: any key/provider error triggers rotation
                last_exc = e
                continue
        raise RuntimeError(f"All {attempts} x.ai API keys failed") from last_exc


_rotator: KeyRotator | None = None
_rotator_lock = threading.Lock()


def get_rotator() -> KeyRotator:
    global _rotator
    if _rotator is None:
        with _rotator_lock:
            if _rotator is None:
                raw = os.environ.get("XAI_API_KEYS", "")
                keys = [k.strip() for k in raw.split(",") if k.strip()]
                if not keys:
                    raise RuntimeError(
                        "XAI_API_KEYS is not set (expected comma-separated keys in project/.env)"
                    )
                _rotator = KeyRotator(keys)
    return _rotator
