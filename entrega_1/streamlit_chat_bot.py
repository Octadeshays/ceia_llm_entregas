import streamlit as st
from RAGDocQA import RAGDocQA
from DocDataBase import DocDataBase
from constants import PINECONE, GROQ

# Initialize the DocDataBase and RAGDocQA
doc_path = "docs/"
pinecone_api_key = PINECONE
index_name = "test-docs"
groq_api_key = GROQ

db = DocDataBase(doc_path=doc_path, pinecone_api_key=pinecone_api_key, index_name=index_name)
rag_chat = RAGDocQA(doc_database=db, groq_api_key=groq_api_key)

# Streamlit App Interface
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Title
st.title("RAG Chatbot with Pinecone and Groq")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Display
st.markdown("### Chat")
chat_placeholder = st.container()  # Placeholder for the chat history


# Input Section
question = st.text_input("Ask your question:")


# Submit Button
if st.button("Send"):
    if question.strip():
        # Answer the question
        result = rag_chat.answer_question(question)
        
        # Save to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": result['answer']})
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
