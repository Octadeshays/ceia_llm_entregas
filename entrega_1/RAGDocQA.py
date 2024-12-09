import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from groq import Groq
from entrega_1.DocDataBase import DocDataBase
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

class RAGDocQA:
    def __init__(self, doc_database: DocDataBase, groq_api_key, model_name="llama3-8b-8192"):
        """
        Initializes the RAG pipeline using Groq's Llama3 API.

        Args:
            doc_database (DocDataBase): An instance of the `DocDataBase` class.
            groq_api_key (str): Groq API key.
            llama_model_name (str): Llama3 model name available on Groq. Defaults to "llama3-8b-8192".
        """
        self.doc_db = doc_database
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = model_name

        self.context_instruction="Use the provided context to answer the question. Answers the questions in third person using the provided information, you are not the person the context talks about"
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
    
    def _generate_answer(self, question, context, chat_history):
        """
        Generates an answer using Groq's Llama3-based API, incorporating chat history.

        Args:
            question (str): The user's question.
            context (str): The retrieved context for answering the question.
            chat_history (str): The chat history to provide additional context.

        Returns:
            str: The generated answer from the Llama3 model.
        """
        # Prepare the prompt, including chat history
        prompt = (
            f"You are a helpful assistant. You are not the person described in the context. Use the provided context and chat history to answer the question.\n\n"
            f"Chat History:\n{chat_history}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )

        # Make a call to the Groq LLM
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content

    def answer_question(self, question):
        """
        Answers a question using the RAG pipeline with Groq's Llama3.

        Args:
            question (str): The question to answer.
            context_instruction (str): Instruction for how to use the retrieved context. Defaults to a generic prompt.

        Returns:
            dict: A dictionary containing the answer and source documents.
        """
        # Retrieve relevant documents
        retrieved_docs = self.doc_db.query_data(query=question)

        # Combine context from retrieved documents
        combined_context = "\n".join([doc for doc in retrieved_docs])

        # Retrieve chat history as a string
        chat_history = "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
            for msg in self.memory.chat_memory.messages
        ])

        # Generate answer using Groq's Llama3 API with chat history
        answer = self._generate_answer(question, combined_context, chat_history)

        # Save the interaction to memory
        self.memory.save_context({"question": question}, {"answer": answer})

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "chat_history": self.get_chat_history()
        }

    def get_chat_history(self):
        """
        Retrieves the current conversation history.

        Returns:
            list: The conversation history as a list of messages.
        """
        return self.memory.load_memory_variables({})['chat_history']
    
    def clear_chat_history(self):
        """
        Clears the current conversation history.
        """
        self.memory.clear()
