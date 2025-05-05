# Master Thesis RAG



## Sprint 1

Added initital vertex base generation and retrieval logic with langchain, huggingface and FAISS.
Retrieved text is just plain extraction and not adjusted for LLM usage.

Chunking is currently simple with a fixed character amount and some overlap.

---

Added LLM integration with Ollama. Currently only via code and not in a conversation. Utilizing RetriavalQA from langchain as RAG pipeline.
