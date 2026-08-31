"""Run the OpenAI Agents SDK pipeline once and print the call ledger.

This file carries the orchestration that adk_version expresses declaratively as
ParallelAgent and SequentialAgent. There is no equivalent object here — it is
asyncio.gather and an f-string, and that contrast is the point of the repo.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

from agents import Runner

from openai_version.agent import (
    MCP_SERVERS,
    budget_agent,
    destination_agent,
    itinerary_agent,
)
from shared import settings, trace


async def run_pipeline(prompt: str) -> str:
    """Drive the three agents once and return the itinerary text."""
    # MCP_SERVERS is empty under TRIP_TOOLS=local, so this costs nothing there.
    # Under mcp it is one MCPServerStdio per agent, and each wraps a subprocess
    # that must stay alive for the whole run -- this SDK makes the caller own
    # that lifecycle, where ADK's McpToolset hides it entirely. An ExitStack
    # keeps the count open-ended instead of hard-coding two `async with` lines.
    async with contextlib.AsyncExitStack() as stack:
        for server in MCP_SERVERS:
            await stack.enter_async_context(server)

        # ADK's ParallelAgent, spelled out: two independent runs, gathered. The
        # blocking httpx calls inside the tools don't stall the loop — the SDK
        # offloads sync tools via asyncio.to_thread.
        destination, budget = await asyncio.gather(
            Runner.run(destination_agent, prompt),
            Runner.run(budget_agent, prompt),
        )

        # And SequentialAgent's {placeholder} injection, spelled out: the briefs are
        # just strings we splice into the next agent's input.
        itinerary_input = (
            f"ORIGINAL REQUEST:\n{prompt}\n\n"
            f"DESTINATION RESEARCH:\n{destination.final_output}\n\n"
            f"BUDGET RESEARCH:\n{budget.final_output}"
        )
        itinerary = await Runner.run(itinerary_agent, itinerary_input)
        return itinerary.final_output


async def main(prompt: str) -> None:
    trace.configure_logging()
    trace.reset()

    print(f"prompt: {prompt}")
    print(f"config: {settings.describe()}")

    itinerary = await run_pipeline(prompt)
    trace.print_report("OPENAI AGENTS SDK", itinerary)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "5 days in Lisbon, I hold USD"))
