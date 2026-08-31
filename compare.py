"""Run one prompt through both SDKs and print the two ledgers side by side.

This is the whole point of the repo: same tools, same prompts, same model, same
retry policy — so anything that differs below is the SDK.
"""

from __future__ import annotations

import asyncio
import sys

from adk_version.run import run_pipeline as adk_pipeline
from openai_version.run import run_pipeline as openai_pipeline
from shared import settings, trace


async def _timed(pipeline, prompt: str):
    """Run one pipeline in isolation and return (itinerary, ledger snapshot).

    trace.CALLS is a single global list shared by both SDKs, so each run has to
    be bracketed by reset-then-snapshot. Skip that and the second run appends
    onto the first, and both ledgers read as one long run.
    """
    trace.reset()
    try:
        itinerary = await pipeline(prompt)
    except Exception as exc:                 # one side failing shouldn't hide
        itinerary = f"FAILED: {trace.explain(exc)}"   # the other. explain()
        #        digs the real cause out of ADK's TaskGroup ExceptionGroup
    return itinerary, trace.snapshot()       # the other side's ledger


async def main(prompt: str) -> None:
    trace.configure_logging()

    print(f"prompt: {prompt}")
    print(f"config: {settings.describe()}")

    adk_text, adk_calls = await _timed(adk_pipeline, prompt)

    # Two full pipelines inside one minute exceed Groq's 8000 tokens/min, and the
    # 429 lands partway through the second half. See settings.COOLDOWN_SECONDS.
    if settings.COOLDOWN_SECONDS:
        print(f"\n(cooling down {settings.COOLDOWN_SECONDS:g}s so the second run "
              "starts in a fresh rate-limit window)")
        await asyncio.sleep(settings.COOLDOWN_SECONDS)

    openai_text, openai_calls = await _timed(openai_pipeline, prompt)

    trace.print_report("GOOGLE ADK", adk_text, adk_calls)
    trace.print_report("OPENAI AGENTS SDK", openai_text, openai_calls)

    adk_counts = trace.call_counts(adk_calls)
    openai_counts = trace.call_counts(openai_calls)
    print("\n" + "=" * 64)
    print("DIFF")
    print("=" * 64)
    if adk_counts == openai_counts:
        print(f"  tool counts identical: {adk_counts}")
    else:
        for name in sorted(set(adk_counts) | set(openai_counts)):
            a, o = adk_counts.get(name, 0), openai_counts.get(name, 0)
            flag = "  " if a == o else "<-"
            print(f"  {flag} {name:<24} adk={a}  openai={o}")
    print(f"  order  adk:    {[c.name for c in adk_calls]}")
    print(f"  order  openai: {[c.name for c in openai_calls]}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "5 days in Lisbon, I hold USD"))
