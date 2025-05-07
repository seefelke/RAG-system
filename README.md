# Master Thesis RAG



## Sprint 1

Added initital vertex base generation and retrieval logic with langchain, huggingface and FAISS.
Retrieved text is just plain extraction and not adjusted for LLM usage.

Chunking is currently simple with a fixed character amount and some overlap.

---

Added LLM integration with Ollama. Currently only via code and not in a conversation. Utilizing RetriavalQA from langchain as RAG pipeline.

---

Added continuous chatting with LLM and refactored single query into seperate function.

Currently, it sometimes answers in english and answers too rigid based on the PDF. E.g. a query that has no connection to the PDF like "Antworte nur in Deutsch" does not really work. ConversationBufferMemory seems to be deprecated so changing that might help.

## Comments from Daksitha
- instead of  ConversationBufferMemory maybe you could give a try 
    ````memory = ConversationBufferWindowMemory(
            k=10,  # number of conversation turns (or messages) to keep
            memory_key="chat_history",
            return_messages=True
        )
         together with ConversationalRetrievalChain

---
Tried ConversationBufferWindowMemory but it didn't improve it, but maybe I used it wrong.

Now I switched the chatting structure to use create_retrieval_chain with premade prompt templates and system context for the agent.

This improved the agent to stick to German as well as taking previous context into account, although it still works poorly when talking about things that are not part of the document.