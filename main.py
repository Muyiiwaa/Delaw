from utils import get_auto_merging_engine, get_auto_merging_index, get_documents
import streamlit as st
import time


index_dir = "./automerging_index_3"
am_index_3 = get_auto_merging_index(get_documents(), index_dir, chunk_sizes=[2048, 512, 128])
am_engine_3 = get_auto_merging_engine(am_index_3)



def response_generator(prompt):
    window_response_1 = am_engine_3.query(prompt)
    response = window_response_1.response
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("CHAT WITH DELAW")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt=prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})