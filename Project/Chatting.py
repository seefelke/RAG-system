from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

import Extraction
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage

chat_model = "mistral"
llm = ChatOllama(model=chat_model)
retriever = Extraction.get_vectorstore().as_retriever()

# Reformulates the current user question based on chat history if needed to give history context
# (query, conversation history) -> LLM -> rephrased query -> retriever -> LLM
history_prompt = (
    "Gegeben sind ein Chatverlauf und die letzte Nutzerfrage,"
    "die sich möglicherweise auf den Kontext im Chatverlauf bezieht."
    "Formuliere eine eigenständige Frage, die auch ohne den Chatverlauf verständlich ist."
    "Beantworte die Frage NICHT, sondern formuliere sie nur um, falls nötig,"
    "und gib sie andernfalls unverändert zurück."
)

history_template = ChatPromptTemplate(
    [
        ("system", history_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# System prompt to answer the actual (reformulated) user query

system_prompt = (
    "Du bist ein Helfer um Fragen in einem Museum zu beantworten. "
    "Verwende den folgenden zusätzlichen Kontext um deine Antwort zu verbessern."
    "Wenn der Kontext nicht zu der Frage passt und du nicht antworten kannst"
    "dann sag dass du dabei nicht helfen kannst."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, history_template
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

rag_chain_simple = create_retrieval_chain(retriever, question_answer_chain)

# Alternative system

prompt_template = PromptTemplate(
            template=(
                "Du bist ein Helfer um Fragen in einem Museum zu beantworten. "
                "Antworte im Dialog kurz und präzise in gesprochener Sprache und versuche dich auf wenige Sätze zu "
                "beschränken."
                "Antworte immer nur auf deutsch"
                "Nutze den folgenden Kontext zur Museums Ausstellung um die Fragen zu beantworten.\n\n"
                "Kontext:\n{context}\n\n"
                "Verlauf:\n{chat_history}\n\n"
                "Frage:\n{question}\n\n"
            ),
            input_variables=["chat_history", "context", "question"]
        )


memory = ConversationBufferWindowMemory(
    k=5,  # Number of conversation turns (or messages) to keep
    memory_key="chat_history",
    return_messages=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)


def continuous_chatting(rag_chain):
    while True:
        chat_history = []
        query = input("\nStelle eine Frage (oder 'exit'): ")
        if query.lower() == "exit":
            break
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        #result = qa_chain.invoke({"question": query,
        #                               "chat_history": memory.chat_memory.messages})
        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=result["answer"]),
            ])
        print("\nAntwort:", result["answer"])


def single_Query(query):
    llm = ChatOllama(model="mistral")
    retriever = Extraction.get_vectorstore().as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke(query)
    return result["result"]
