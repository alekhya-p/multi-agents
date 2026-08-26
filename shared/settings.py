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
        # three agents can share one model. Must support tool calling.
        "models": (lambda m: (m, m, m))(
            os.getenv("TRIP_GROQ_MODEL", "openai/gpt-oss-120b")
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


def describe() -> str:
    """One line for run output, so a ledger is never read without its config."""
    mode = "fast-fail" if FAST_FAIL else "patient"
    models = {DESTINATION_MODEL_NAME, BUDGET_MODEL_NAME, ITINERARY_MODEL_NAME}
    shown = models.pop() if len(models) == 1 else (
        f"{DESTINATION_MODEL_NAME}/{BUDGET_MODEL_NAME}/{ITINERARY_MODEL_NAME}"
    )
    extra = f"  overrides={_ACTIVE_OVERRIDES}" if _ACTIVE_OVERRIDES else ""
    return (
        f"provider={PROVIDER}  models={shown}  retries={RETRY_ATTEMPTS} extra={extra}  "
        f"backoff={RETRY_INITIAL_DELAY:g}-{RETRY_MAX_DELAY:g}s  [{mode}]"
    )