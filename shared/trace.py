"""Call tracing shared by both SDK implementations.

The point of this project is to SEE what each framework does, so every tool call
is recorded in one global, ordered ledger. Both versions import this same module,
which makes their call sequences directly comparable.
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("trip")


def configure_logging(level: int = logging.INFO) -> None:
    """Readable single-line logs. Safe to call more than once."""
    # Windows consoles default to cp1252 and raise on any character outside it —
    # arrows, degree signs, em dashes, all of which models emit constantly. This
    # sits ABOVE the early return below on purpose: a repeat call must still fix
    # the streams, or a run dies in the final print after succeeding.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

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
    # ADK runs its own function-calling loop over generate_content, so the genai
    # SDK warns on every call that we should be using AsyncChat.send_message
    # instead. It's ADK-internal and not actionable from here — silence it.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


@dataclass
class Call:
    seq: int
    name: str
    ok: bool = True
    error: str | None = None
    duration_ms: float = 0.0


# Ordered ledger of every traced call in this process.
CALLS: list[Call] = []

# Both SDKs run tools concurrently — ADK via ParallelAgent, the OpenAI SDK via
# asyncio.gather plus asyncio.to_thread for sync tools. Numbering and appending
# are two steps, so without this two tools can claim the same seq.
_LOCK = threading.Lock()


def reset() -> None:
    """Clear the ledger. Call between runs so counts stay meaningful."""
    with _LOCK:
        CALLS.clear()


def snapshot() -> list[Call]:
    """A copy of the ledger as it stands, for comparing runs after a reset()."""
    with _LOCK:
        return list(CALLS)


def call_counts(calls: list[Call] | None = None) -> dict[str, int]:
    """How many times each tool ran — the idempotency signal."""
    counts: dict[str, int] = {}
    for c in CALLS if calls is None else calls:
        counts[c.name] = counts.get(c.name, 0) + 1
    return counts


def format_sequence(calls: list[Call] | None = None) -> str:
    """Human-readable replay of the run."""
    calls = CALLS if calls is None else calls
    if not calls:
        return "  (no calls recorded)"
    return "\n".join(
        f"  {c.seq:>2}. {'OK  ' if c.ok else 'FAIL'} {c.name:<24} "
        f"{c.duration_ms:>6.0f}ms  {c.error or ''}"
        for c in calls
    )


def traced(fn: Callable) -> Callable:
    """Record every invocation of `fn` in the ledger, success or failure."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _LOCK:                          # number and append as one step, so
            record = Call(seq=len(CALLS) + 1, name=fn.__name__)
            CALLS.append(record)             # append BEFORE running, so crashes
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


def explain(exc: BaseException) -> str:
    """Flatten an exception down to the messages that actually say something.

    ADK's MCP session manager runs on an anyio TaskGroup, and anything that
    fails inside one surfaces as::

        ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)

    which names no cause at all -- the real error ("timed out after 5.0s
    waiting for the session to become ready") is a leaf inside it. Walk to the
    leaves so a failed run reports something actionable.
    """
    leaves: list[str] = []

    def walk(e: BaseException) -> None:
        subs = getattr(e, "exceptions", None)
        if subs:
            for sub in subs:
                walk(sub)
        else:
            leaves.append(f"{type(e).__name__}: {e}")

    walk(exc)
    # dict.fromkeys dedupes while keeping order -- a TaskGroup often carries the
    # same failure several times, once per task it cancelled.
    return " | ".join(dict.fromkeys(leaves)) or f"{type(exc).__name__}: {exc}"


def print_report(label: str, itinerary: str, calls: list[Call] | None = None) -> None:
    """Print one run's itinerary and ledger. Shared so both versions and the
    comparator format results identically — a diff should show behaviour, not
    layout."""
    bar = "=" * 64
    print(f"\n{bar}\n{label}\n{bar}")
    print(itinerary.strip() or "(no itinerary returned)")
    print("\n--- call ledger ---")
    print(format_sequence(calls))
    print(f"  counts: {call_counts(calls)}")
