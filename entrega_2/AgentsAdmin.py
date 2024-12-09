import os
import sys

# Add the entrega_1 folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'entrega_1'))

# Import all modules from the entrega_1 folder
from entrega_1.RAGDocQA import RAGDocQA
from entrega_1.DocDataBase import DocDataBase
from entrega_1.constants import PINECONE, GROQ


class AgentsAdmin:

    def __init__(self, docs_path: str, pinecone_api_key: str, groq_api_key: str):
        self.rag_agents = {}
        
        for file_name in os.listdir(docs_path):
            doc_path = os.path.join(docs_path, file_name)
            index_name = file_name
            print(f"Initializing agent for {file_name} with index name {index_name}")
            db = DocDataBase(doc_path=doc_path, pinecone_api_key=pinecone_api_key, index_name=index_name)
            self.rag_agents[file_name] = RAGDocQA(doc_database=db, groq_api_key=groq_api_key)
        
        self.selected_agent_name = None

    def answer_question(self, selected_agent ,question: str):
        return self.rag_agents[selected_agent].answer_question(question)
