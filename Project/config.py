import os
from langchain_openai import OpenAIEmbeddings

USE_OPENAI = True
USE_FAISS = False
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = "mistral"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 80
if USE_FAISS:
    STORE_TYPE = "FAISS"
else:
    STORE_TYPE = "PINECONE"
if USE_OPENAI:
    EMBEDDINGS = OpenAIEmbeddings()
else:
    EMBEDDINGS = "sentence-transformers/distiluse-base-multilingual-cased-v2"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if USE_OPENAI:
    INDEX_NAME = "openai-index-museum-thesis"
else:
    INDEX_NAME = "langchain-index-museum-thesis"
MODELS = ["Mistral", "GPT 4o Mini"]
EXTRA_NOTES = "-"