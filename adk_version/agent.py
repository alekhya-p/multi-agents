"""The trip-planning pipeline on Google ADK.

Orchestration is declarative here: ParallelAgent and SequentialAgent are real
objects, and the briefs move between agents through session state. Compare
with openai_version/run.py, where the same shape is asyncio.gather and an
f-string.
"""

from dotenv import load_dotenv

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models import Gemini
from google.genai import types

from shared import settings
from shared.apis import geocode_city, get_weather, get_wikipedia_summary, get_exchange_rate

load_dotenv()

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
    # call, so retry and timeout config carries over.
    from google.adk.models.lite_llm import LiteLlm
    return LiteLlm(
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
    tools=[geocode_city, get_weather, get_wikipedia_summary],
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
    tools=[get_exchange_rate],
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
        "DESTINATION RESEARCH:\n{destination_brief}\n\n"
        "BUDGET RESEARCH:\n{budget_brief}\n\n"
        "Give a day-by-day outline that reflects the weather, then a budget note."
    ),
)

root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[research_phase, itinerary_agent],
)
