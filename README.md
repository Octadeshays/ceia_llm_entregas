# LLM e IA Generativa - Universidad de Buenos Aires



# ENTREGA 1:RAG Chatbot with Pinecone and Groq

## Overview
This is a **Retrieval-Augmented Generation (RAG) Chatbot** application that integrates:
- **Pinecone** for efficient document retrieval.
- **Groq's LLM** for generating answers based on retrieved context.
- A **Streamlit-based interface** for an interactive and user-friendly chat experience.

The app processes `.odt` documents, indexes them, and uses a combination of retrieval and language modeling to provide accurate, context-aware answers to user queries.

---

## Features
1. **Document Upload & Retrieval**:
   - Reads `.odt` documents from the `docs/` folder.
   - Splits the content into manageable chunks and indexes them in Pinecone.

2. **Conversational Chat Interface**:
   - Displays questions and answers in a chat-like format.
   - Maintains chat history to provide continuity in conversations.

3. **Retrieval-Augmented Generation (RAG)**:
   - Retrieves relevant context from the indexed documents using Pinecone.
   - Generates answers via Groq's LLM by incorporating retrieved context and chat history.

---

## Requirements
### Prerequisites
- **Python 3.9+**
- API keys:
  - Pinecone API Key
  - Groq API Key

### Install Required Packages
Use the following command to install the dependencies:
```bash
pip install requirements.txt
