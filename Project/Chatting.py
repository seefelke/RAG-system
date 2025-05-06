import Extraction
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

llm = ChatOllama(model="mistral")
retriever = Extraction.get_vectorstore().as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, memory=memory)

while True:
    query = input("\nStelle eine Frage (oder 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa_chain.invoke(query)
    print("\nAntwort:", result["result"])


def single_Query(query):
    llm = ChatOllama(model="mistral")
    retriever = Extraction.get_vectorstore().as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke(query)
    return result["result"]
