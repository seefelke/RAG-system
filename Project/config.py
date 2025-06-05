import os
from langchain_openai import OpenAIEmbeddings

USE_OPENAI = True
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
STORE_TYPE = "FAISS"
if USE_OPENAI:
    EMBEDDINGS = OpenAIEmbeddings()
else:
    EMBEDDINGS = "sentence-transformers/distiluse-base-multilingual-cased-v2"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "langchain-index"