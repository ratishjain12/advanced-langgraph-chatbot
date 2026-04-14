from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)
checkpointer = InMemorySaver()

def chat_node(state: ChatState) -> ChatState:
    user_message = state['messages']

    response = llm.invoke(user_message)

    return {"messages": [response]}


graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


    
