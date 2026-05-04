from langgraph.graph import START, StateGraph
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
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

_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}


@tool
async def rag_search(query: str) -> str:
    """
    Search indexed PDF documents for information relevant to the user's question.
    
    Use this tool when the user asks questions about content from previously uploaded 
    PDF documents in the current conversation thread. This tool performs semantic search
    across all indexed document chunks to find the most relevant information.
    
    Args:
        query: The search query string. Formulate a question or keyword phrase that captures
              what information you're looking for from the documents.
    
    Returns:
        A string containing the top matching document excerpts with source context.
        Each excerpt is followed by a reference to its source document.
    
    When to use:
        - User asks "what does the document say about X?"
        - User asks "can you summarize the contents of the PDF?"
        - User asks a question that likely refers to previously uploaded documents
        - User wants to find specific information from a PDF they uploaded
        
    When NOT to use:
        - User asks general knowledge questions (use duckduckgo_search instead)
        - User asks math calculations (use calculator tool)
        - No PDF has been uploaded yet in this conversation thread
    """
    runtime = get_runtime()
    config = runtime and runtime.config
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    if not thread_id:
        return "No conversation thread context available."
    
    retriever = _THREAD_RETRIEVERS.get(str(thread_id))
    if not retriever:
        return "No PDF documents have been indexed in this conversation. Please upload a PDF first using the ingest button in the sidebar."
    
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant information found in the indexed documents for your query."
    
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        results.append(f"[{i}] Source: {source}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)


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
    tools = [search_tool, rag_search] + mcp_tools

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


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str | None = None) -> dict:
    """Build a FAISS index from the provided PDF file bytes and store it under thread_id."""
    import os
    import tempfile
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings

    if not file_bytes:
        raise ValueError("No file provided")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(tmp_file_path),
            "documents": len(documents),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(tmp_file_path),
            "documents": len(documents),
            "chunks": len(chunks),
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")
    finally:
        try:
            os.remove(tmp_file_path)
        except Exception as e:
            print(f"Warning: Failed to delete temporary file: {e}")


def retrieve_all_threads() -> list[str]:
    async def _list():
        threads: set[str] = set()
        async for checkpoint in checkpointer.alist(None):
            threads.add(checkpoint.config["configurable"]["thread_id"])
        return list(threads)

    return _run_async(_list())


def get_thread_state(config: dict):
    return _run_async(chatbot.aget_state(config))
