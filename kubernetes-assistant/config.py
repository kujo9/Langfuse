#"""Everything this app reads from the environment, in one place."""

import os

from dotenv import load_dotenv
from langfuse import get_client

# Copies the .env next to this file into the process environment.
# os.getenv below reads our settings back out of it, and the Anthropic
# library finds ANTHROPIC_API_KEY there on its own: we never pass it.
load_dotenv()

# --- model ---
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
STEP_LIMIT = int(os.getenv("STEP_LIMIT", "30"))  # how many times the agent may loop before giving up

# --- cluster ---
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8080/mcp")
NAMESPACE = os.getenv("NAMESPACE", "default")

# --- observability ---
USER_ID = os.getenv("USER_ID", "student")
PROMPT_NAME = "k8s-system"
PROMPT_LABEL = os.getenv("PROMPT_LABEL", "production")
DATASET_NAME = "k8s-questions"

# The live prompt is in Langfuse. This copy is what seed_langfuse.py uploads,
# and what the app falls back to when Langfuse can't be reached.
SYSTEM_PROMPT = """
You are a read-only Kubernetes assistant. The default namespace is {{namespace}}.

Gather evidence before you conclude. Look at namesopaces, pods, then events, then logs.
A pod can report phase Running while its container sits in CrashLoopBackOff,
so read the waiting reason and the restart count, not the phase alone.

Report only what a tool returned. If a tool returns nothing, say so.
If a fix needs a write, describe the command and do not run it.
"""


def system_prompt():
    """The k8s-system prompt from Langfuse, or the copy above when Langfuse can't be reached."""
    # No cache: a cached prompt is served stale once more after it expires, so an
    # edit in Langfuse would only reach the second question asked after it.
    # Fetching every time costs a few milliseconds against the local Langfuse.
    # If Langfuse is down, give up after 3 seconds and use the copy above.
    return get_client().get_prompt(
        PROMPT_NAME, label=PROMPT_LABEL, fallback=SYSTEM_PROMPT,
        cache_ttl_seconds=0, fetch_timeout_seconds=3, max_retries=0,
    )