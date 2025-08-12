import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

USE_OPENAI = False
USE_FAISS = True
USE_HISTORY = False
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNK_AMOUNT = 8
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
VECTORSTORES = ["FAISS", "PINECONE"]
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
    "Du bist ein Helfer in einem Museum und beantwortest Fragen im Dialog mit einem Besucher."
    "Nutze den bereitgestellten Kontext, der auf Ausstellungstexten und Informationen zu Exponaten basiert, "
    "um deine Antworten zu verbessern."
    " Antworte stets kurz und präzise – verwende nicht mehr als 50 Wörter."
    " Formuliere deine Antworten in fließendem, grammatikalisch korrektem Deutsch."
    " Vermeide nummerierte Stichpunkte, Bulletpoints oder andere Aufzählungsformen."
    " Gib stattdessen vollständige Sätze in einem zusammenhängenden Fließtext wieder."
    "\n\n"
    "Context: {context}"
)
