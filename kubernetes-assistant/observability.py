"""Langfuse for the whole app: one client, secrets masked, automatic scores."""
import re

import config  # noqa: F401  loads .env before the client reads its keys
from langfuse import Langfuse, is_langfuse_span

# Pod logs and env output can contain credentials. Every input, output and
# metadata value passes through mask() before it leaves this process.
SECRETS = [
    (re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)(\s*[=:]\s*)\S+"), r"\1\2[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-(?:ant|lf)-[A-Za-z0-9_-]+"), "[REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}"), "[REDACTED]"),
]


def mask(*, data, **kwargs):
    """Replace anything that looks like a secret. Langfuse calls this on every value."""
    if isinstance(data, str):
        for pattern, replacement in SECRETS:
            data = pattern.sub(replacement, data)
        return data
    if isinstance(data, dict):
        return {key: mask(data=value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(data=value) for value in data]
    return data


# Created once, before anything asks for a client: get_client() anywhere
# in the app returns this same masked instance.
# is_langfuse_span keeps only Langfuse's own spans. Without it, the Anthropic
# SDK's built-in tracing records every Claude call a second time.
langfuse = Langfuse(mask=mask, should_export_span=is_langfuse_span)

# Tool names that change the cluster. With read_only = true the server
# doesn't offer them, so a call to one means something is misconfigured.
WRITE_WORDS = ("create", "update", "delete", "patch", "scale", "exec", "run")


def is_write_tool(name):
    return any(word in name for word in WRITE_WORDS)


def score_step_limit(hit):
    """1 when the agent ran out of steps. Call inside the k8s-ask trace."""
    langfuse.score_current_trace(name="hit_step_limit", value=1 if hit else 0, data_type="BOOLEAN")


def score_tool_use(messages):
    """Automatic checks on the tool calls behind one answer. Call inside the k8s-ask trace."""
    results = [m for m in messages if m.type == "tool"]
    checks = {
        "tool_calls": len(results),
        "tool_errors": sum(getattr(m, "status", "success") == "error" for m in results),
        "empty_tool_results": sum(not m.text.strip() for m in results),
    }
    for name, value in checks.items():
        langfuse.score_current_trace(name=name, value=value, data_type="NUMERIC")
    wrote = any(is_write_tool(m.name) for m in results)
    langfuse.score_current_trace(name="write_attempted", value=1 if wrote else 0, data_type="BOOLEAN")


def score_feedback(trace_id, thumbs_up):
    """Thumbs from the page. score_id makes a changed vote replace the old one."""
    langfuse.create_score(
        trace_id=trace_id,
        name="user_feedback",
        value=1 if thumbs_up else 0,
        data_type="BOOLEAN",
        score_id=f"{trace_id}-user-feedback",
    )