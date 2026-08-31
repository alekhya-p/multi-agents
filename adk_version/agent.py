"""The trip-planning pipeline on Google ADK.

Orchestration is declarative here: ParallelAgent and SequentialAgent are real
objects, and the briefs move between agents through session state. Compare
with openai_version/run.py, where the same shape is asyncio.gather and an
f-string.
"""

import asyncio
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from litellm.exceptions import RateLimitError

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from shared import settings, trace
from shared.apis import geocode_city, get_weather, get_wikipedia_summary, get_exchange_rate

load_dotenv()

# ------------------------------------------------------------------- tools

_LOCAL_TOOLS = {
    "geocode_city": geocode_city,
    "get_weather": get_weather,
    "get_wikipedia_summary": get_wikipedia_summary,
    "get_exchange_rate": get_exchange_rate,
}

_SERVER_PY = str(Path(__file__).resolve().parent.parent / "mcp_server.py")


def trip_tools(*names: str) -> list:
    """The named tools, imported locally or fetched over MCP per TRIP_TOOLS.

    tool_filter is not optional. mcp_server.py advertises all four, while
    destination_agent has only ever seen three and budget_agent one. Handing an
    agent tools it never had would move the ledger for a reason that has nothing
    to do with MCP -- exactly the kind of fake finding shared/settings.py exists
    to prevent.
    """
    if settings.TOOLS == "local":
        return [_LOCAL_TOOLS[name] for name in names]

    # Imported lazily so the local path never needs `mcp` installed, and so an
    # MCP import error can't break a run that wasn't using MCP.
    from mcp import StdioServerParameters
    from google.adk.tools.mcp_tool import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

    # One toolset == one server SUBPROCESS. Two agents means two servers, and
    # shared.trace.CALLS lives in whichever process the tool RAN in -- so the
    # ledger this side prints comes back empty under TRIP_TOOLS=mcp. That is the
    # process boundary, not a bug.
    return [
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,      # the venv python running us
                    args=[_SERVER_PY],
                ),
                timeout=settings.MCP_SESSION_TIMEOUT,
            ),
            tool_filter=list(names),
        )
    ]


# --------------------------------------------------------------- rate limits

_RETRY_AFTER = re.compile(r"try again in ([0-9.]+)s", re.I)


class PatientLiteLlm(LiteLlm):
    """LiteLlm that actually waits out a rate limit.

    LiteLLM caps its own rate-limit backoff at 10s -- litellm/main.py:5831 is a
    hard-coded tenacity.wait_exponential(multiplier=1, max=10) -- and ignores
    both the Retry-After header and settings.RETRY_MAX_DELAY. Groq's free tier
    meters 8000 tokens/minute and routinely asks for 20-40s, so num_retries
    expires long before the window clears and the ADK run dies.

    The OpenAI side has no such problem: AsyncOpenAI honours Retry-After. So
    without this the two SDKs fail differently under load, and a comparison
    would be measuring LiteLLM's retry policy rather than the frameworks.
    """

    async def generate_content_async(self, llm_request, stream: bool = False):
        delay = settings.RETRY_INITIAL_DELAY
        for attempt in range(1, settings.RETRY_ATTEMPTS + 1):
            produced = False
            try:
                async for response in super().generate_content_async(llm_request, stream):
                    produced = True
                    yield response
                return
            except RateLimitError as exc:
                # Never retry once something has been yielded -- the caller has
                # already seen part of the answer and would get it twice.
                if produced or attempt == settings.RETRY_ATTEMPTS:
                    raise
                asked = _RETRY_AFTER.search(str(exc))
                wait = float(asked.group(1)) + 1 if asked else delay
                wait = min(wait, settings.RETRY_MAX_DELAY)
                trace.log.warning(
                    "rate limited, waiting %.1fs (attempt %d/%d)",
                    wait, attempt, settings.RETRY_ATTEMPTS,
                )
                await asyncio.sleep(wait)
                delay = min(delay * settings.RETRY_EXP_BASE, settings.RETRY_MAX_DELAY)


def build_model(model_name: str):
    """Native Gemini where we can, LiteLLM for everything else."""
    if settings.PROVIDER == "gemini":
        return Gemini(
            model=model_name,
            retry_options=types.HttpRetryOptions(
                attempts=settings.RETRY_ATTEMPTS,
                initial_delay=settings.RETRY_INITIAL_DELAY,
                max_delay=settings.RETRY_MAX_DELAY,
                exp_base=settings.RETRY_EXP_BASE,
                jitter=settings.RETRY_JITTER,
                http_status_codes=settings.RETRY_STATUS_CODES,
            ),
        )
    # LiteLlm(model, **kwargs) forwards kwargs straight to litellm's completion
    # call, so retry and timeout config carries over -- except the rate-limit
    # backoff, which is why PatientLiteLlm exists.
    return PatientLiteLlm(
        model=f"{settings.LITELLM_PREFIX}{model_name}",
        api_key=settings.api_key(),
        num_retries=settings.RETRY_ATTEMPTS,
        timeout=settings.REQUEST_TIMEOUT,
    )

# Model per agent, resolved by provider in shared/settings.py — three distinct
# models under Gemini (its free tier meters 20/day PER MODEL), one shared model
# under Groq. Both SDKs read the same names, which is what keeps the comparison
# honest.
DESTINATION_MODEL = build_model(settings.DESTINATION_MODEL_NAME)
BUDGET_MODEL      = build_model(settings.BUDGET_MODEL_NAME)
ITINERARY_MODEL   = build_model(settings.ITINERARY_MODEL_NAME)

destination_agent = LlmAgent(
    name="destination_agent",
    model=DESTINATION_MODEL,
    tools=trip_tools("geocode_city", "get_weather", "get_wikipedia_summary"),
    description="on search of city it gets latitude, longtitude and search for summary for city",
    instruction=(
        "You search for travel destination \n"
        "1. First call geocode_city with the input of the user to get latitude, longtitude and country \n"
        "2. Next call get_weather using latitude, longtitude \n"
        "3. At last call get_wikipedia_summary with city name \n"
        "Then write the short brief"
    ),
    output_key="destination_brief",
)

budget_agent = LlmAgent(
    name="budget_agent",
    model=BUDGET_MODEL,
    tools=trip_tools("get_exchange_rate"),
    description="Converts the currency rate",
    instruction=(
        "The user's message names a destination city and the currency they hold.\n"
        "Infer the destination country from the city, and its currency from the "
        "country (Lisbon -> Portugal -> EUR).\n"
        "Call get_exchange_rate(base=<currency they hold>, target=<destination currency>).\n"
        "If no held currency is given, assume USD and say so.\n"
        "Write one short line with the rate."
    ),
    output_key="budget_brief",
)

research_phase = ParallelAgent(
    name="research_phase",
    sub_agents=[destination_agent, budget_agent],
)

# {destination_brief} and {budget_brief} are filled from session state, written
# by the two output_keys above. This injection is what the OpenAI version has
# to do by hand.
itinerary_agent = LlmAgent(
    name="itinerary_agent",
    model=ITINERARY_MODEL,
    instruction=(
        "Write a trip plan using only the research below.\n\n"
        "Give a day-by-day outline that reflects the weather, then a budget note."
    ),
)

root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[research_phase, itinerary_agent],
)
