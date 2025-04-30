from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load(path):
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=650,
                                                   chunk_overlap=70,
                                                   length_function=len,
                                                   is_separator_regex=False)
    return text_splitter.split_documents(documents)

def get_vectorstore() -> VectorStore:
    path = "PDF"
    documents = load(path)

    chunks = split_documents(documents)
    model = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    embedding = HuggingFaceEmbeddings(model_name=model)
    return FAISS.from_documents(chunks, embedding)