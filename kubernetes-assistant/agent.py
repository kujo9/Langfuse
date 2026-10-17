"""The agent: a model, some tools, a loop. Every question becomes one Langfuse trace."""

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError

import observability  # creates the masked Langfuse client before anything else uses one
from config import MCP_URL, MODEL, NAMESPACE, STEP_LIMIT, USER_ID, system_prompt
from tools import load_tools

model = ChatAnthropic(model=MODEL)
langfuse = get_client()


@observe(name="k8s-ask", capture_input=False, capture_output=False)
async def ask(question, history=(), session_id=None):
    """Ask one question. Returns the answer, the tools the model called, and the trace id."""
    # The question alone is this span's input, and so the trace's input.
    langfuse.update_current_span(input=question)
    prompt = system_prompt()

    # Everything inside this block, LangChain's steps included, gets these attributes.
    # Metadata keys: letters and digits only. Values: strings of at most 200 characters.
    # prompt= links every Claude call to the prompt version that produced it.
    with propagate_attributes(
        trace_name="k8s-ask",
        user_id=USER_ID,
        session_id=session_id,
        tags=["kubernetes-assistant", f"namespace:{NAMESPACE}"],
        prompt=prompt,
        metadata={
            "model": MODEL,
            "namespace": NAMESPACE,
            "stepLimit": str(STEP_LIMIT),
            "mcpUrl": MCP_URL,
            "promptVersion": str(prompt.version),
            "promptFallback": str(prompt.is_fallback),
        },
    ):
        tools = await load_tools()
        agent = create_agent(model, tools, system_prompt=prompt.compile(namespace=NAMESPACE))

        # Only role and content go to the model; the page also stores trace ids.
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": question})

        try:
            result = await agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": STEP_LIMIT, "callbacks": [CallbackHandler()]},
            )
        except GraphRecursionError:
            answer = f"I stopped after {STEP_LIMIT} steps without reaching an answer. Try a narrower question."
            observability.score_step_limit(hit=True)
            langfuse.update_current_span(
                output=answer,
                level="ERROR",
                status_message=f"Hit STEP_LIMIT={STEP_LIMIT}",
                # The Step 8 judges need this key on every run, failed ones included.
                metadata={"evidence": "No final answer: the agent hit the step limit."},
            )
            return answer, [], langfuse.get_current_trace_id()

        called = [m.name for m in result["messages"] if m.type == "tool"]
        answer = result["messages"][-1].text
        # What each tool returned, shortened, so the LLM judges in Step 8 can check the answer against it.
        evidence = "\n\n".join(f"[{m.name}]\n{m.text[:2000]}" for m in result["messages"] if m.type == "tool")

        observability.score_step_limit(hit=False)
        observability.score_tool_use(result["messages"])
        langfuse.update_current_span(
            output=answer,
            metadata={
                "toolsOffered": ", ".join(sorted(t.name for t in tools)),
                "toolsCalled": ", ".join(called),
                "evidence": evidence[:20000] or "No tool was called.",
            },
        )
        return answer, called, langfuse.get_current_trace_id()


async def tool_names():
    """The tool names the server offers. Change read_only in mcp-config.toml to compare."""
    return sorted(t.name for t in await load_tools())