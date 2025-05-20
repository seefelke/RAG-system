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


09/05/2025 

    ````python 
        self.memory = ConversationBufferWindowMemory(
        k=0,  # number of conversation turns (or messages) to keep
        memory_key="chat_history",
        return_messages=True
    )
    self.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0.8, api_key=openai_api_key),
        retriever=retriever,
        memory=self.memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )





    11:00
    result = self.qa_chain.invoke({"question": query,
                                "chat_history": self.memory.chat_memory.messages})

TODO: 
- Clean data in: check the parsed document
- Is it needed to have two chat templates?
    ```
    prompt_template = PromptTemplate(
                template=(
                    "You are a helpful assistant named Cora. You appear as an avatar at the 'Alles Fake? Täuschend echt or echt getäuscht' exhibition at the Museum Oberschönenfeld (from April 6th to October 12th, 2025). "
                    "Please always answer concisely and in a spoken style, and keep your answer under 200 characters. "
                    "Use the following context — extracted from the exhibition statement — to accurately answer the following question.\n\n"
                    "Context:\n{context}\n\n"
                    "History:\n{chat_history}\n\n"
                    "Question:\n{question}\n\n"
                    "Answer:"
                ),
                input_variables=["chat_history", "context", "question"]
            )
- Understand how chattemplate is parsed to the models.
- Creation of ground-truth questions and answers. Compare them with generated answers. 
- Literature review: https://dl.acm.org/doi/pdf/10.1145/3708359.3712145

---

### Sprint 2

Loaded PDF is now saved back to disc as txt file to check for inconsistencies.

Added a questions and answers json for evaluation. Questions have 4 categories:

Simple questions with a short answer.

Question pairs that consist of two questions that have the same answer but are phrased in a simple and a difficult way.

Difficult questions that require a longer answer and more context.

General questions that have no connection to the PDF.

---

Added second QA chain like in the above example for testing. 

Quality of answers is similiar to the previous version but for some reason it rarely switched to english or answered in broken German.

One thing to note is when asking about something general it often quotes something random from the document and then adds the answer to the question at the end.

Memory is hit or miss in both versions, sometimes it works great and sometimes it answers something random or too general.