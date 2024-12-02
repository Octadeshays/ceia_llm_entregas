import os
from dotenv import load_dotenv
load_dotenv()



PINECONE = os.getenv('PINECONE')
GROQ = os.getenv('GROQ')