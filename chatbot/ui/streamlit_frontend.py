import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot.backend.langgraph_backend import chatbot
import uuid

# utilities
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_thread(thread_id):
    st.session_state['thread_id'] = thread_id
    return chatbot.get_state(config={'configurable': {
        'thread_id': thread_id
    }}).values['messages']

# streamlit ui
def main():

    # session management
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []
    
    if 'thread_id' not in st.session_state:
        st.session_state['thread_id'] = generate_thread_id()

    if "chat_threads" not in st.session_state:
        st.session_state['chat_threads'] = []

    add_thread(st.session_state['thread_id'])

    # sidebar
    st.sidebar.title("Advance Langgraph Chatbot")

    if st.sidebar.button("New Chat"):
        reset_chat()

    st.sidebar.header("My Conversations")

    for thread_id in st.session_state['chat_threads'][::-1]:
        if st.sidebar.button(str(thread_id)):
            messages = load_thread(thread_id)
            temp_messages = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    role = 'user'
                else: 
                    role = 'assistant'
                temp_messages.append({'role': role, 'content': message.content})
            st.session_state['message_history'] = temp_messages
            

    # chat ui
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.text(message['content'])

    user_input = st.chat_input('Type here')

    if user_input:
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message("user"):
            st.text(user_input)

        with st.chat_message("assistant"):
            ai_message = st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config={'configurable': {
                        'thread_id': st.session_state['thread_id']
                    }},
                    stream_mode= "messages"
                )
            )
            st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})