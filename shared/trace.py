"""Call tracing shared by both SDK implementations.

The point of this project is to SEE what each framework does, so every tool call
is recorded in one global, ordered ledger. Both versions import this same module,
which makes their call sequences directly comparable.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("trip")


def configure_logging(level: int = logging.INFO) -> None:
    """Readable single-line logs. Safe to call more than once."""
    root = logging.getLogger()
    if root.handlers:          # already configured — don't double up
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-5s  %(name)-12s  %(message)s",
                          datefmt="%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)
    # These are extremely chatty and would drown out our own trace.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)


@dataclass
class Call:
    seq: int
    name: str
    ok: bool = True
    error: str | None = None
    duration_ms: float = 0.0


# Ordered ledger of every traced call in this process.
CALLS: list[Call] = []


def reset() -> None:
    """Clear the ledger. Call between runs so counts stay meaningful."""
    CALLS.clear()


def call_counts() -> dict[str, int]:
    """How many times each tool ran — the idempotency signal."""
    counts: dict[str, int] = {}
    for c in CALLS:
        counts[c.name] = counts.get(c.name, 0) + 1
    return counts


def format_sequence() -> str:
    """Human-readable replay of the run."""
    if not CALLS:
        return "  (no calls recorded)"
    return "\n".join(
        f"  {c.seq:>2}. {'OK  ' if c.ok else 'FAIL'} {c.name:<24} "
        f"{c.duration_ms:>6.0f}ms  {c.error or ''}"
        for c in CALLS
    )


def traced(fn: Callable) -> Callable:
    """Record every invocation of `fn` in the ledger, success or failure."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        record = Call(seq=len(CALLS) + 1, name=fn.__name__)
        CALLS.append(record)                 # append BEFORE running, so crashes
        log.info("-> %s", fn.__name__)       # still appear in the right position
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            record.duration_ms = (time.perf_counter() - started) * 1000
            log.warning("<- %s FAILED: %s", fn.__name__, record.error)
            raise                            # never swallow — the SDK must see it
        record.duration_ms = (time.perf_counter() - started) * 1000
        log.info("<- %s ok (%.0fms)", fn.__name__, record.duration_ms)
        return result

    return wrapper