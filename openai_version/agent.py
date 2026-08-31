"""The same trip-planning pipeline, on the OpenAI Agents SDK.

Same tools, same prompts, same model and retry policy as adk_version — so
anything that differs in the ledger is the SDK, not the setup.

Note what's missing compared to adk_version/agent.py: there is no ParallelAgent,
no SequentialAgent, and no output_key. This SDK has no orchestration primitives,
so that wiring lives in run.py as plain asyncio.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled

from shared import settings
from shared.apis import (
    geocode_city,
    get_exchange_rate,
    get_weather,
    get_wikipedia_summary,
)

load_dotenv()

_client = AsyncOpenAI(
    base_url=settings.BASE_URL,
    api_key=settings.api_key(),
    max_retries=settings.RETRY_ATTEMPTS,
    timeout=settings.REQUEST_TIMEOUT,
)

# Trace upload goes to platform.openai.com and needs an OpenAI key we don't
# have. shared/trace.py is the ledger this project actually compares.
set_tracing_disabled(True)


def model(name: str) -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(model=name, openai_client=_client)


DESTINATION_MODEL = model(settings.DESTINATION_MODEL_NAME)
BUDGET_MODEL      = model(settings.BUDGET_MODEL_NAME)
ITINERARY_MODEL   = model(settings.ITINERARY_MODEL_NAME)

# --------------------------------------------------------------------- tools

# shared/apis.py stays SDK-neutral — it must never import `agents`, or
# adk_version would pull this SDK in too. So wrap by CALLING, not decorating.
#
# failure_error_function=None means a raised ApiError aborts the run, which is
# what ADK does. Drop the argument and this SDK instead hands the error text
# back to the model and lets it retry — a real fork worth measuring later.
_LOCAL_TOOLS = {
    "geocode_city":          function_tool(geocode_city,          failure_error_function=None),
    "get_weather":           function_tool(get_weather,           failure_error_function=None),
    "get_wikipedia_summary": function_tool(get_wikipedia_summary, failure_error_function=None),
    "get_exchange_rate":     function_tool(get_exchange_rate,     failure_error_function=None),
}

_SERVER_PY = str(Path(__file__).resolve().parent.parent / "mcp_server.py")

# Every MCPServerStdio built below, in creation order. Unlike ADK's McpToolset,
# these do NOT manage their own lifecycle: each one is an async context manager
# wrapping a live subprocess, and run.py has to hold them open for the duration
# of the run. Same protocol, different ownership model — worth recording as a
# genuine difference between the two SDKs. Empty when TRIP_TOOLS=local.
MCP_SERVERS: list = []


def tool_kwargs(*names: str) -> dict:
    """Agent() keyword arguments for the named tools, per TRIP_TOOLS.

    The filter is not optional. mcp_server.py advertises all four, while
    destination_agent has only ever seen three and budget_agent one. Handing an
    agent tools it never had would move the ledger for a reason that has nothing
    to do with MCP -- exactly the kind of fake finding shared/settings.py exists
    to prevent.
    """
    if settings.TOOLS == "local":
        return {"tools": [_LOCAL_TOOLS[name] for name in names]}

    # Imported lazily so the local path never needs `mcp` installed.
    from agents.mcp import MCPServerStdio, create_static_tool_filter

    server = MCPServerStdio(
        params={"command": sys.executable, "args": [_SERVER_PY]},
        tool_filter=create_static_tool_filter(allowed_tool_names=list(names)),
        cache_tools_list=True,        # don't re-issue tools/list every turn
        name=f"trip-tools[{','.join(names)}]",
    )
    MCP_SERVERS.append(server)
    return {"mcp_servers": [server]}


destination_agent = Agent(
    name="destination_agent",
    model=DESTINATION_MODEL,
    **tool_kwargs("geocode_city", "get_weather", "get_wikipedia_summary"),
    instructions=(
        "You search for travel destination \n"
        "1. First call geocode_city with the input of the user to get latitude, longtitude and country \n"
        "2. Next call get_weather using latitude, longtitude \n"
        "3. At last call get_wikipedia_summary with city name \n"
        "Then write the short brief"
    ),
)

budget_agent = Agent(
    name="budget_agent",
    model=BUDGET_MODEL,
    **tool_kwargs("get_exchange_rate"),
    instructions=(
        "The user's message names a destination city and the currency they hold.\n"
        "Infer the destination country from the city, and its currency from the "
        "country (Lisbon -> Portugal -> EUR).\n"
        "Call get_exchange_rate(base=<currency they hold>, target=<destination currency>).\n"
        "If no held currency is given, assume USD and say so.\n"
        "Write one short line with the rate."
    ),
)

# No output_key and no {placeholder} here. ADK routes the two briefs through
# session state; this SDK has no state to route through, so run.py holds them
# in variables and formats them into the input itself.
itinerary_agent = Agent(
    name="itinerary_agent",
    model=ITINERARY_MODEL,
    instructions=(
        "Write a trip plan using only the research you are given.\n"
        "Give a day-by-day outline that reflects the weather, then a budget note."
    ),
)
