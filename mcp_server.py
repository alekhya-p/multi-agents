"""An MCP server exposing the trip-planning tools.
The same four functions the agents already call, server over the Model Context
Protocol instaed of imported as Python. Any MCP client can use them - Claude code, the MCP Inspector, or (next step) either SDK's agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# A clint launches this from its OWN working directory, so the repo woot won't
#be on sys.path and 'shared' would be unimportable. Fix it here rather than
#depending on how we happen to be invoked.

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from shared import trace
from shared.apis import (
    geocode_city,
    get_exchange_rate,
    get_weather,
    get_wikipedia_summary,
)

# On stdio, STDOUT IS THE PROTOCOL CHANNEL. A stray print() corrupts the JSON-RPC
# stream and the client drops the connection. Logging goes to stderr, which is
# safe and surfaces in the client's server log. (configure_logging also forces
# UTF-8 on both streams, which MCP wants anyway.)
 
trace.configure_logging()

mcp = FastMCP("trip-tools")
 # Register by calling, not decorating, so shared/apis.py never imports `mcp`.
 # The docstrings there already document every argument, and that text becomes
 # the description the model sees — same as under ADK and the OpenAI SDK.
 
for fn in (geocode_city,get_weather, get_wikipedia_summary, get_exchange_rate):
    mcp.tool()(fn)
    
if __name__ == "__main__":
    mcp.run()
 