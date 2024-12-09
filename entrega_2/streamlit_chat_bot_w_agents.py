import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'entrega_1')))

# Import all modules from the entrega_1 folder
from entrega_1.constants import PINECONE, GROQ
from AgentsAdmin import AgentsAdmin

# Initialize the DocDataBase and RAGDocQA
doc_path = "entrega_2/docs/"
pinecone_api_key = PINECONE
groq_api_key = GROQ

if "agents_admin" not in st.session_state:
    st.session_state.agents_admin = AgentsAdmin(docs_path=doc_path, pinecone_api_key=pinecone_api_key, groq_api_key=groq_api_key)
agents_amdmin = st.session_state.agents_admin

# Streamlit App Interface
st.set_page_config(page_title="Agents Chatbot", layout="wide")

# Title
st.title("RAG Chatbot with Pinecone and Groq")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar for CV selection
with st.sidebar:
    st.session_state.selected_cv = st.selectbox("Select a Candidate's CV", list(agents_amdmin.rag_agents.keys()))

# Chat Display
st.markdown("### Chat")
chat_placeholder = st.container()  # Placeholder for the chat history


# Detect if selected_cv has changed
if "previous_selected_cv" not in st.session_state:
    st.session_state.previous_selected_cv = None

if st.session_state.selected_cv != st.session_state.previous_selected_cv:
    st.session_state.previous_selected_cv = st.session_state.selected_cv
    st.session_state.chat_history = []  # Reset chat history when CV changes

if st.session_state.selected_cv:
    # Input Section
    question = st.text_input("Ask your question:")

    # Submit Button
    if st.button("Send"):
        if question.strip():
            # Answer the question
            result = agents_amdmin.answer_question(selected_agent=st.session_state.selected_cv, question=question)
            
            # Save to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": result['answer']})
            question = ""
        else:
            st.warning("Please enter a question before sending.")

    # Always refresh the chat display after user interaction
    with chat_placeholder:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f"<div style='text-align: left; background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin: 5px;'>"
                    f"<strong>User:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='text-align: right; background-color: #dcedc8; padding: 10px; border-radius: 10px; margin: 5px;'>"
                    f"<strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)
else:
    st.warning("Please select a valid CV before asking a question.")
