from dotenv import load_dotenv

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from shared.apis import geocode_city, get_weather, get_wikipedia_summary, get_exchange_rate
from google.adk.models import Gemini
from google.genai import types
load_dotenv()



def with_retry(model_name: str) -> Gemini:
    """Same model, but retries 429s with exponential backoff instead of dying."""
    return Gemini(
        model=model_name,
        retry_options=types.HttpRetryOptions(
            attempts=5,
            initial_delay=8.0,    # free tier resets per minute, so start high
            max_delay=70.0,       # long enough to cross a full quota window
            exp_base=2.0,
            jitter=0.3,           # stagger parallel agents so they don't sync up
            http_status_codes=[429, 500, 502, 503, 504],
        ),
    )
    
DESTINATION_MODEL  = with_retry("gemini-3.6-flash")
BUDGET_MODEL    = with_retry("gemini-3.5-flash")
ITINERARY_MODEL = with_retry("gemini-flash-latest")

# MODEL = "gemini-2.5-flash"
destination_agent = LlmAgent(
    name="destination_agent",
    model=DESTINATION_MODEL,
    tools=[geocode_city, get_weather, get_wikipedia_summary],
    description="on search of city it gets latitude, longtitude and search for summary for city",
    instruction=(
        "You search for travel destination \n"
        "1. First call geocode_city with the input of the user to get latitude, longtitude and country \n"
        "2. Next call get_weather using latitude, longtitude \n"
        "3. At last call get_wikipedia_summary with country name \n"
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
        "You convert the currency rate \n"
        "Should infer the destination currency from the country"
        "write the converstion rate"
    ),
    output_key="budget_brief"
)

research_phase = ParallelAgent(
    name="research_phase",
    sub_agents=[destination_agent, budget_agent]
)
itinerary_agent = LlmAgent(
    name=ITINERARY_MODEL,
    model="gemini-flash-latest",
    instruction=(
         "Write a trip plan using only the research below.\n\n"
        "DESTINATION RESEARCH:\n{destination_brief}\n\n"
        "BUDGET RESEARCH:\n{budget_brief}\n\n"
        "Give a day-by-day outline that reflects the weather, then a budget note."
    )
)

root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[research_phase, itinerary_agent]
)

