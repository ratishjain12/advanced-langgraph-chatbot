import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chatbot.backend.langgraph_backend import chatbot, retrieve_all_threads, stream_graph, get_thread_state, ingest_pdf
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
    state = get_thread_state(config={'configurable': {
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

    st.sidebar.header("Ingest PDF")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.sidebar.button("Index Document"):
            with st.spinner("Indexing document..."):
                try:
                    result = ingest_pdf(
                        uploaded_file.read(),
                        st.session_state['thread_id'],
                        uploaded_file.name
                    )
                    st.sidebar.success(f"Indexed: {result['filename']} ({result['chunks']} chunks)")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

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
            st.markdown(message['content'])

    user_input = st.chat_input('Type here')

    if user_input:
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {'configurable': {
                        'thread_id': st.session_state['thread_id']
                    },
                   'metadata': {
                        "thread_id": st.session_state['thread_id']
                    },
                    'run_name': "chat_turn"
                }

        with st.chat_message("assistant"):
            # tool_placeholder created FIRST → renders ABOVE answer text
            tool_placeholder = st.empty()
            answer_placeholder = st.empty()

            full_response = ""
            # tool_info[index] = {"id": str, "name": str, "args_str": str}
            tool_info = {}
            # Maps tool_call_id -> index for ToolMessage lookup
            id_to_index = {}
            # Indices already shown as "running" (avoid duplicate renders)
            rendered_running = set()

            for message_chunk, metadata in stream_graph(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
            ):
                node = metadata.get("langgraph_node", "")

                # ── chat_node: accumulate tool_call_chunks or stream text ────
                if node == "chat_node" and isinstance(message_chunk, AIMessage):
                    tc_chunks = getattr(message_chunk, "tool_call_chunks", [])

                    if tc_chunks:
                        for tc in tc_chunks:
                            idx = tc.get("index", 0)
                            if idx not in tool_info:
                                tool_info[idx] = {"id": "", "name": "", "args_str": ""}
                            # id and name arrive complete in the very first chunk
                            if tc.get("id"):
                                tool_info[idx]["id"] = tc["id"]
                                id_to_index[tc["id"]] = idx
                            if tc.get("name"):
                                tool_info[idx]["name"] = tc["name"]
                            # args are streamed char-by-char — accumulate them
                            tool_info[idx]["args_str"] += tc.get("args", "")

                            # Render "running" status ONCE when name is known
                            if idx not in rendered_running and tool_info[idx]["name"]:
                                rendered_running.add(idx)
                                with tool_placeholder.container():
                                    with st.status(
                                        f"🔧 Calling: **{tool_info[idx]['name']}**",
                                        state="running",
                                        expanded=True,
                                    ):
                                        st.markdown("_Waiting for tool result..._")

                    elif message_chunk.content:
                        # Plain LLM text token — stream into answer_placeholder
                        full_response += message_chunk.content
                        answer_placeholder.markdown(full_response + "▌")

                # ── tools node: update placeholder to complete with input+output
                elif node == "tools" and isinstance(message_chunk, ToolMessage):
                    tc_id = getattr(message_chunk, "tool_call_id", "")
                    tool_output = message_chunk.content

                    idx = id_to_index.get(tc_id, next(iter(tool_info), 0))
                    info = tool_info.get(idx, {})
                    tool_name = info.get("name") or getattr(message_chunk, "name", "Tool")

                    # Parse the fully-accumulated args JSON
                    try:
                        args = json.loads(info.get("args_str", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                    # Replace the running status with completed: input + output
                    with tool_placeholder.container():
                        with st.status(f"🔧 **{tool_name}**", state="complete", expanded=True):
                            st.markdown("**📥 Input:**")
                            st.json(args)
                            st.divider()
                            st.markdown("**📤 Output:**")
                            st.markdown(tool_output)

            # Finalise — remove blinking cursor
            answer_placeholder.markdown(full_response)
            st.session_state['message_history'].append(
                {'role': 'assistant', 'content': full_response}
            )