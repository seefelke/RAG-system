import Extraction
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

llm = ChatOllama(model="mistral")
retriever = Extraction.get_vectorstore().as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "Welche Strategien verwenden Tiere um das Überleben zu sichern?"
result = qa_chain.invoke(query)

print(result["query"])
print("-------------------------------------------------------------------")
print(result["result"])