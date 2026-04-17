from langgraph.graph import START, StateGraph
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import sqlite3

load_dotenv()

# Define tools
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Bind tools to the LLM
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)

conn = sqlite3.connect(database = 'chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

def chat_node(state: ChatState) -> dict:
    """Process user message and potentially request tool calls."""
    user_message = state['messages']
    response = llm_with_tools.invoke(user_message)
    return {"messages": [response]}

# Create tool node with error handling
tool_node = ToolNode(tools, handle_tool_errors=True)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "chat_node")
# Use tools_condition to route: if LLM requested tools -> "tools", otherwise -> END
graph.add_conditional_edges("chat_node", tools_condition)
# After tools execute, go back to chat_node for LLM to process results
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


    
