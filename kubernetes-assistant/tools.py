"""Where the tools come from."""
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse import get_client, observe

from config import MCP_URL


@observe(name="mcp-load-tools", capture_output=False)
async def load_tools():
    """Connect to the MCP server and return the tools it offers."""
    client = MultiServerMCPClient({"k8s": {"transport": "streamable_http", "url": MCP_URL}})
    tools = await client.get_tools()
    get_client().update_current_span(output=sorted(t.name for t in tools))
    return tools