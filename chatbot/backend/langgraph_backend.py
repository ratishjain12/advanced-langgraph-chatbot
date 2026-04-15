from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)

conn = sqlite3.connect(database = 'chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

def chat_node(state: ChatState) -> ChatState:
    user_message = state['messages']

    response = llm.invoke(user_message)

    return {"messages": [response]}


graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


    
