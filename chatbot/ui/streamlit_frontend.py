import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot.backend.langgraph_backend import chatbot, retrieve_all_threads
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
    state = chatbot.get_state(config={'configurable': {
        'thread_id': thread_id
    }})

    if 'messages' in state.values:
        return state.values['messages']
    return []

# streamlit ui
def main():

    # session management
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    if "chat_threads" not in st.session_state:
        st.session_state['chat_threads'] = retrieve_all_threads()

    if 'thread_id' not in st.session_state:
        if st.session_state['chat_threads']:
            # Restore the most recent existing thread instead of creating a new one
            most_recent_thread = st.session_state['chat_threads'][-1]
            st.session_state['thread_id'] = most_recent_thread
            st.session_state['message_history'] = load_thread(most_recent_thread)
            # Convert LangChain messages to display format
            temp_messages = []
            for message in st.session_state['message_history']:
                role = 'user' if isinstance(message, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': message.content})
            st.session_state['message_history'] = temp_messages
        else:
            # No threads exist yet — create the first one
            st.session_state['thread_id'] = generate_thread_id()
            st.session_state['chat_threads'].append(st.session_state['thread_id'])

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

        config = {'configurable': {
                        'thread_id': st.session_state['thread_id']
                    },
                   'metadata': {
                        "thread_id": st.session_state['thread_id']
                    },
                    'run_name': "chat_turn"
                }

        with st.chat_message("assistant"):
            ai_message = st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config= config,
                    stream_mode= "messages"
                )
            )
            st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})