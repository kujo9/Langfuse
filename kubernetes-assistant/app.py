import asyncio
import atexit
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from streamlit import runtime
from streamlit.web import cli as stcli

# `python app.py` works too: without Streamlit's server around us, start one
# on this same file. `streamlit run app.py` skips this and goes straight on.
if __name__ == "__main__" and not runtime.exists():
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())

import agent
import config
import observability
from langfuse import get_client

HERE = Path(__file__).parent
SERVER = ["uvx", "kubernetes-mcp-server@latest", "--config", "mcp-config.toml"]
WAIT_SECONDS = 120       # the first run downloads the server, so be patient

EXAMPLES = [
    "Why is checkout-service unavailable?",
    "Show me any pods that are restarting.",
    "Delete the unhealthy pod.",
]

st.set_page_config(page_title="Kubernetes Assistant", page_icon="K8s", layout="centered")
st.title("Kubernetes Assistant")
st.caption(f"Read only, namespace {config.NAMESPACE}. It never changes the cluster.")


# --- MCP server ---
# `streamlit run app.py` starts the Kubernetes MCP server too, so one terminal
# is enough. Streamlit reruns this file on every click; cache_resource makes
# the start happen once per Streamlit process, shared by every browser session.

def port_open():
    url = urlparse(config.MCP_URL)
    try:
        with socket.create_connection((url.hostname, url.port), timeout=1):
            return True
    except OSError:
        return False


def stop_server(server):
    if server.poll() is not None:
        return
    # Kill the whole tree: uvx starts the server as a child of its own.
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
    else:
        os.killpg(server.pid, 15)
    server.wait()


@st.cache_resource(show_spinner="Starting the Kubernetes MCP server...")
def start_server():
    """Start the server unless one already answers at MCP_URL."""
    if port_open():
        print(f"Using the MCP server already running at {config.MCP_URL}")
        return None

    # A process group of its own, so Ctrl+C reaches Streamlit, which then stops the server.
    if os.name == "nt":
        server = subprocess.Popen(SERVER, cwd=HERE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        server = subprocess.Popen(SERVER, cwd=HERE, start_new_session=True)
    atexit.register(stop_server, server)

    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"The MCP server exited with code {server.returncode}. See the terminal.")
        if port_open():
            return server
        time.sleep(0.5)
    stop_server(server)
    raise RuntimeError(f"The MCP server did not answer at {config.MCP_URL} within {WAIT_SECONDS} seconds.")


try:
    start_server()
except RuntimeError as error:
    st.error(str(error))
    st.stop()


@st.cache_resource
def start_langfuse():
    """Check the keys once per Streamlit process, and send buffered traces when it stops."""
    client = get_client()
    atexit.register(client.shutdown)
    try:
        return client.auth_check()
    except Exception:
        return False


if not start_langfuse():
    st.warning("Langfuse is not reachable, so this chat is not traced. See Troubleshooting in the lab guide.")


# Each chat is a list of messages. They live in this browser session only,
# so every engineer sees their own chats and a page refresh clears them.
if "chats" not in st.session_state:
    st.session_state.chats = [[]]
    st.session_state.chat_ids = [str(uuid.uuid4())]
    st.session_state.current = 0


def new_chat():
    if st.session_state.chats[st.session_state.current]:
        st.session_state.chats.append([])
        st.session_state.chat_ids.append(str(uuid.uuid4()))
        st.session_state.current = len(st.session_state.chats) - 1


def open_chat(index):
    st.session_state.current = index


def send_feedback(trace_id):
    """Runs when a thumb is clicked: 1 is up, 0 is down, None is unclicked."""
    value = st.session_state[f"feedback-{trace_id}"]
    if value is not None:
        observability.score_feedback(trace_id, thumbs_up=value == 1)


def feedback(message):
    if message.get("trace_id"):
        st.feedback("thumbs", key=f"feedback-{message['trace_id']}", on_change=send_feedback, args=(message["trace_id"],))


messages = st.session_state.chats[st.session_state.current]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            feedback(message)

if not messages:
    st.write("Try one of these:")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.pending = example
            st.rerun()

question = st.chat_input("Ask about your cluster")

if not question and "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Investigating..."):
            session_id = st.session_state.chat_ids[st.session_state.current]
            answer, called, trace_id = asyncio.run(agent.ask(question, messages[:-1], session_id))

        if called:
            st.caption("Tools called: " + ", ".join(called))
        st.markdown(answer)
        message = {"role": "assistant", "content": answer, "trace_id": trace_id}
        feedback(message)

    messages.append(message)

# The sidebar holds previous chats and nothing else. It is drawn last so a
# chat started in this run is already listed.
with st.sidebar:
    st.button("New chat", on_click=new_chat, use_container_width=True)
    st.caption("Previous chats")
    for index in reversed(range(len(st.session_state.chats))):
        chat = st.session_state.chats[index]
        if chat:
            st.button(
                chat[0]["content"][:40],
                key=f"chat-{index}",
                on_click=open_chat,
                args=(index,),
                use_container_width=True,
                type="primary" if index == st.session_state.current else "secondary",
            )