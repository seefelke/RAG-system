import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

USE_OPENAI = False
USE_FAISS = True
USE_HISTORY = False
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = "mistral"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
if USE_FAISS:
    STORE_TYPE = "FAISS"
else:
    STORE_TYPE = "PINECONE"
if USE_OPENAI:
    MODEL_NAME = "GPT 4o Mini"
    EMBEDDINGS = "OpenAI Embeddings"
else:
    EMBEDDINGS = "sentence-transformers/distiluse-base-multilingual-cased-v2"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if USE_OPENAI:
    INDEX_NAME = "openai-index-museum-thesis"
else:
    INDEX_NAME = "langchain-index-museum-thesis"
MODELS = ["Mistral", "GPT 4o Mini"]
EMBEDDING_SELECTION = ["sentence-transformers/distiluse-base-multilingual-cased-v2"]
EXTRA_NOTES = "-"

HISTORY_PROMPT = (
    "Gegeben sind ein Chatverlauf und die letzte Nutzerfrage,"
    "die sich möglicherweise auf den Kontext im Chatverlauf bezieht."
    "Formuliere eine eigenständige Frage, die auch ohne den Chatverlauf verständlich ist."
    "Beantworte die Frage NICHT, sondern formuliere sie nur um, falls nötig,"
    "und gib sie andernfalls unverändert zurück."
)

# System prompt to answer the actual (reformulated) user query

SYSTEM_PROMPT = (
    "Du bist ein Helfer um Fragen in einem Museum zu beantworten. "
    "Verwende den folgenden zusätzlichen Kontext um deine Antwort zu verbessern."
    "Wenn der Kontext nicht zu der Frage passt und du nicht antworten kannst"
    "dann sag dass du dabei nicht helfen kannst."
    "\n\n"
    "Context: {context}"
)