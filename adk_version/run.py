"""Run the ADK pipeline once and print the call ledger.

`adk web` is the nicer way to poke at this interactively, but it can't emit a
ledger — this file exists so the ADK side is comparable with the OpenAI side.
"""

from __future__ import annotations

import asyncio
import sys

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_version.agent import root_agent
from shared import settings, trace

APP_NAME = "trip_planner"
USER_ID = "learner"


async def run_pipeline(prompt: str) -> str:
    """Drive root_agent once and return the itinerary text."""
    # A place for agents to leave notes for each other. InMemory, not the
    # .adk/session.db the CLI writes — a script run shouldn't inherit UI history.
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    # Every one of these is keyword-only in ADK 2.7; older tutorials pass
    # new_message positionally and fail here.
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    # is_final_response() is True once PER AGENT, not once per run — three times
    # here, since the sequence is [destination, budget] then itinerary. So keep
    # overwriting rather than breaking; the last one is itinerary_agent.
    # (It has no output_key, so session.state would hold the two briefs but
    # never the itinerary itself.)
    final = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(part.text or "" for part in event.content.parts)

    return final


async def main(prompt: str) -> None:
    trace.configure_logging()
    trace.reset()

    print(f"prompt: {prompt}")
    print(f"config: {settings.describe()}")

    itinerary = await run_pipeline(prompt)
    trace.print_report("GOOGLE ADK", itinerary)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "5 days in Lisbon, I hold USD"))
