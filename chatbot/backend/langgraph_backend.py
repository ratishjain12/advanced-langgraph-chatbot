from langgraph.graph import START, StateGraph
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

import asyncio
import threading
import queue
import sys

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Persistent background event loop — all async work dispatched here
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="bg-async-loop").start()


def _run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


_mcp_client: MultiServerMCPClient = None  # type: ignore
_checkpointer_ctx = None   # keeps the context manager alive
checkpointer: AsyncSqliteSaver = None     # type: ignore


async def _init_graph_async():
    global _mcp_client, checkpointer, _checkpointer_ctx

    _checkpointer_ctx = AsyncSqliteSaver.from_conn_string("chatbot.db")
    checkpointer = await _checkpointer_ctx.__aenter__()

    _mcp_client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["chatbot/mcp/math.py"],
            }
        }
    )
    mcp_tools = await _mcp_client.get_tools()

    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool] + mcp_tools

    llm = ChatOpenAI(streaming=True)
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState) -> dict:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    g = StateGraph(ChatState)
    g.add_node("chat_node", chat_node)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_edge(START, "chat_node")
    g.add_conditional_edges("chat_node", tools_condition)
    g.add_edge("tools", "chat_node")

    return g.compile(checkpointer=checkpointer)


chatbot = _run_async(_init_graph_async())


def stream_graph(input_messages: dict, config: dict):
    """Sync generator wrapping chatbot.astream() for Streamlit."""
    q: queue.Queue = queue.Queue()

    async def _producer():
        try:
            async for item in chatbot.astream(input_messages, config, stream_mode="messages"):
                q.put(item)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(None)

    asyncio.run_coroutine_threadsafe(_producer(), _loop)

    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def retrieve_all_threads() -> list[str]:
    async def _list():
        threads: set[str] = set()
        async for checkpoint in checkpointer.alist(None):
            threads.add(checkpoint.config["configurable"]["thread_id"])
        return list(threads)

    return _run_async(_list())


def get_thread_state(config: dict):
    return _run_async(chatbot.aget_state(config))
