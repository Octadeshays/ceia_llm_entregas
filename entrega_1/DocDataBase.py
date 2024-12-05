from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from odf.opendocument import load
from odf.text import P

from constants import PINECONE


import os
import time
from odf.opendocument import load
from odf.text import P
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

class DocDataBase:
    
    def __init__(self, doc_path, pinecone_api_key, index_name="docs", chunk_size=800, chunk_overlap=50):
        """
        Initializes the DocDataBase class with Pinecone and document handling setup.

        Args:
            doc_path (str): Path to the folder containing .odt files.
            pinecone_api_key (str): API key for Pinecone.
            index_name (str): Name of the Pinecone index. Defaults to "docs".
        """
        self.pinecone = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name

        # Ensure the index exists
        if not self.pinecone.has_index(self.index_name):
            self.pinecone.create_index(
                name=self.index_name,
                dimension=1024,  
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # Wait for the index to be ready
        while not self.pinecone.describe_index(self.index_name).status["ready"]:
            time.sleep(1)

        self.documentos = self.leer_documentos_odt(doc_path)

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunked_doc = self.text_splitter.split_documents(self.documentos)
        self.index = self.upsert_data()

        

    def leer_documentos_odt(self, doc_path):
        """
        Reads all .odt files in a folder and returns a list of langchain Document objects.

        Args:
            doc_path (str): Path to the folder containing .odt files.

        Returns:
            list: A list of langchain Documents with content and metadata.
        """
        documentos = {}
        for archivo in os.listdir(doc_path):
            if archivo.endswith(".odt"):
                ruta_archivo = os.path.join(doc_path, archivo)
                documento = load(ruta_archivo)
                contenido = []

                # Extract text from paragraphs
                for elemento in documento.getElementsByType(P):
                    contenido.append(str(elemento))

                documentos[archivo] = "\n".join(contenido)

        langchain_docs = [
            Document(page_content=contenido, metadata={"source": nombre})
            for nombre, contenido in documentos.items()
        ]
        return langchain_docs

    def upsert_data(self):
        """
        Converts documents into embeddings and upserts them into the Pinecone index.
        """
        # Prepare the data for embeddings
        data = [{"id": f"vec{i+1}", "text": doc.page_content} for i, doc in enumerate(self.chunked_doc)]

        # Generate embeddings
        embeddings = self.pinecone.inference.embed(
            model="multilingual-e5-large",
            inputs=[d["text"] for d in data],
            parameters={"input_type": "passage", "truncate": "END"}
        )

        # Prepare the records for upsert
        records = [
            {
                "id": d["id"],
                "values": e["values"],
                "metadata": {"text": d["text"]}
            }
            for d, e in zip(data, embeddings)
        ]

        # Upsert the records into the index
        index = self.pinecone.Index(self.index_name)
        index.upsert(vectors=records, namespace=f"{self.index_name}-namespace")

        return index

    def query_data(self, query, top_k=3):
        """
        Queries the Pinecone index with a given query string.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to retrieve. Defaults to 3.

        Returns:
            list: A list of matching results with metadata.
        """
        # Convert query to embedding
        query_embedding = self.pinecone.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={"input_type": "query"}
        )

        # Search the index
        results = self.index.query(
            namespace=f"{self.index_name}-namespace",
            vector=query_embedding[0]["values"],
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        docs_list = [document['metadata']['text'] for document in results['matches']]
        return docs_list

    

    
