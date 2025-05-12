import Extraction
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage

llm = ChatOllama(model="mistral")
retriever = Extraction.get_vectorstore().as_retriever()

system_prompt_history = (
    "Gegeben sind ein Chatverlauf und die letzte Nutzerfrage,"
    "die sich möglicherweise auf den Kontext im Chatverlauf bezieht."
    "Formuliere eine eigenständige Frage, die auch ohne den Chatverlauf verständlich ist."
    "Beantworte die Frage NICHT, sondern formuliere sie nur um, falls nötig,"
    "und gib sie andernfalls unverändert zurück."
)
system_prompt = (
    "Du bist ein Helfer um Fragen zu beantworten. "
    "Verwende den folgenden zusätzlichen Kontext um deine Antwort zu verbessern."
    "Wenn der Kontext nicht zu der Frage passt und du nicht antworten kannst"
    "dann sag dass du dabei nicht helfen kannst."
    "\n\n"
    "{context}"
)
q_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt_history),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, q_prompt
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

while True:
    query = input("\nStelle eine Frage (oder 'exit'): ")
    if query.lower() == "exit":
        break
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
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
