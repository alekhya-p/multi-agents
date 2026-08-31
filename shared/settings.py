from __future__ import annotations

import os

# ---------------------------------------------------------------- provider

PROVIDER = os.getenv("TRIP_PROVIDER", "groq")

_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "litellm_prefix": "groq/",
        # Groq's limits aren't punishing per-model the way Gemini's are, so all
        # three agents can share one model. It must support tool calling AND not
        # be a reasoning model: Groq returns `reasoning`, LiteLLM renames it to
        # `reasoning_content` and echoes it back on the next request, and Groq
        # then rejects its own field -- which kills the ADK side on turn two,
        # right after the first tool call. Of the tool-capable models on this
        # account, qwen3.8-27b is the only one that emits no reasoning field.
        "models": (lambda m: (m, m, m))(
            os.getenv("TRIP_GROQ_MODEL", "qwen/qwen3.8-27b")
        ),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GOOGLE_API_KEY",
        "litellm_prefix": "gemini/",
        # Three distinct models on purpose: the free tier meters 20/day PER MODEL,
        # so this is three separate buckets.
        "models": ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash"),
    },
}

if PROVIDER not in _PROVIDERS:
    raise RuntimeError(f"TRIP_PROVIDER={PROVIDER!r} — expected one of {sorted(_PROVIDERS)}")

_p = _PROVIDERS[PROVIDER]
BASE_URL = _p["base_url"]
API_KEY_ENV = _p["key_env"]
LITELLM_PREFIX = _p["litellm_prefix"]

DESTINATION_MODEL_NAME, BUDGET_MODEL_NAME, ITINERARY_MODEL_NAME = (
    os.getenv("TRIP_DESTINATION_MODEL") or _p["models"][0],
    os.getenv("TRIP_BUDGET_MODEL") or _p["models"][1],
    os.getenv("TRIP_ITINERARY_MODEL") or _p["models"][2],
)

_ACTIVE_OVERRIDES = {
    k: v for k in (
        "TRIP_DESTINATION_MODEL", "TRIP_BUDGET_MODEL",
        "TRIP_ITINERARY_MODEL", "TRIP_GROQ_MODEL",
    )
    if (v := os.getenv(k))
}


def api_key() -> str:
    key = os.getenv(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} not set — required for TRIP_PROVIDER={PROVIDER}")
    return key


# ------------------------------------------------------------------- tools

# How the four functions in shared/apis.py reach the agents:
#   local -> imported into the agent's own process, the way this repo started
#   mcp   -> served by mcp_server.py over stdio, in a SEPARATE process
# Both SDKs read this, so a ledger diff is never transport-vs-import by accident.
TOOLS = os.getenv("TRIP_TOOLS", "local")
if TOOLS not in ("local", "mcp"):
    raise RuntimeError(f"TRIP_TOOLS={TOOLS!r} — expected 'local' or 'mcp'")

# Seconds to wait for a freshly spawned MCP server to finish booting. ADK
# defaults to 5.0, which is too tight here: a cold `python mcp_server.py` takes
# ~0.7s idle, but a run spawns two of them WHILE the main process is mid-LLM
# call, and on Windows interpreter startup competes with antivirus scanning.
# Blowing it produces "Failed to create MCP session: timed out after 5.0s",
# which ADK then reports only as "ExceptionGroup: unhandled errors in a
# TaskGroup" -- see trace.explain().
MCP_SESSION_TIMEOUT = float(os.getenv("TRIP_MCP_TIMEOUT", "30"))


# ------------------------------------------------------------------- retry

# Gemini's free tier meters two ways, and both come back as a 429:
#   GenerateRequestsPerMinutePerProjectPerModel  -> 5/min,  clears in ~25-60s
#   GenerateRequestsPerDayPerProjectPerModel     -> 20/day, does not clear today
# The patient defaults ride out the per-minute one; nothing rides out the daily
# one. FAST_FAIL caps backoff at 10s, so it CANNOT clear even a per-minute 429 --
# it is for catching wiring errors in ~6s, not for runs you want to succeed.
FAST_FAIL = os.getenv("TRIP_FAST_FAIL") == "1"

RETRY_ATTEMPTS = 3 if FAST_FAIL else 5
RETRY_INITIAL_DELAY = 2.0 if FAST_FAIL else 8.0
RETRY_MAX_DELAY = 10.0 if FAST_FAIL else 70.0
RETRY_EXP_BASE = 2.0
RETRY_JITTER = 0.3                                # stagger parallel agents
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# Must exceed RETRY_MAX_DELAY, or a request dies mid-backoff.
REQUEST_TIMEOUT = 30.0 if FAST_FAIL else 90.0

# Groq's free tier meters TOKENS PER MINUTE (8000 for qwen3.8-27b; check with
# x-ratelimit-limit-tokens). One full pipeline costs roughly 4-5k, so anything
# that runs BOTH SDKs inside one minute -- compare.py, webui.py -- goes over and
# gets a 429 partway through the second half.
#
# Retries cannot save it. ADK reaches Groq through LiteLLM, and LiteLLM hard-codes
# its rate-limit backoff to tenacity.wait_exponential(multiplier=1, max=10)
# (litellm/main.py:5831) -- capped at 10s regardless of RETRY_MAX_DELAY below,
# while Groq asks for ~18s. So the fix is to not hit the limit: pause between the
# two pipelines instead.
#
# Note this is also a real asymmetry between the two sides. The OpenAI SDK talks
# to Groq through AsyncOpenAI, which honours the Retry-After header properly;
# LiteLLM does not. Worth remembering before reading a timing difference as SDK
# behaviour.
COOLDOWN_SECONDS = float(
    os.getenv("TRIP_COOLDOWN", "0" if FAST_FAIL or PROVIDER != "groq" else "25")
)


def describe() -> str:
    """One line for run output, so a ledger is never read without its config."""
    mode = "fast-fail" if FAST_FAIL else "patient"
    models = {DESTINATION_MODEL_NAME, BUDGET_MODEL_NAME, ITINERARY_MODEL_NAME}
    shown = models.pop() if len(models) == 1 else (
        f"{DESTINATION_MODEL_NAME}/{BUDGET_MODEL_NAME}/{ITINERARY_MODEL_NAME}"
    )
    extra = f"  overrides={_ACTIVE_OVERRIDES}" if _ACTIVE_OVERRIDES else ""
    return (
        f"provider={PROVIDER}  tools={TOOLS}  models={shown}  retries={RETRY_ATTEMPTS}{extra}  "
        f"cooldown={COOLDOWN_SECONDS:g}s  "
        f"backoff={RETRY_INITIAL_DELAY:g}-{RETRY_MAX_DELAY:g}s  [{mode}]"
    )