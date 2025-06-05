from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import jsonlines
from config import *
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS

# subsection headers for JSONL conversion
section_markers = {"Untergruppentext", "Modultext", "Einführungstext"}

def load(path):
    loader = PyPDFDirectoryLoader(path)

    os.makedirs("loaded_docs", exist_ok=True)
    documents = loader.load()

    output_file = "loaded_doc_combined.txt"
    # Combine all document contents
    combined_text = "\n\n".join(
        f"--- Document {i + 1} ---\n{doc.page_content}" for i, doc in enumerate(documents)
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)

    convert_to_JSONL(documents)

    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP,
                                                   length_function=len,
                                                   is_separator_regex=False)
    return text_splitter.split_documents(documents)

def get_vectorstore() -> VectorStore:
    path = "PDF"
    documents = load(path)
    if USE_OPENAI:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
    else:
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDINGS)
    chunks = split_documents(documents)
    if STORE_TYPE == "FAISS":
        vectorstore = FAISS.from_documents(chunks, embedding)
    else:
        vectorstore = PineconeVectorStore.from_documents(chunks, index_name=INDEX_NAME, embedding=embedding)
    return vectorstore

def convert_to_JSONL(documents):

    all_text = "\n".join(doc.page_content for doc in documents)
    lines = all_text.split("\n")
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line in section_markers:
            entry = {"type": line}
            if i + 1 < len(lines):
                entry["title"] = lines[i + 1].strip()
            else:
                entry["title"] = ""

            # collect content until the next marker or end
            content_lines = []
            i += 2  # skip current marker and title
            while i < len(lines) and lines[i].strip() not in section_markers:
                content_lines.append(lines[i].strip())
                i += 1
            entry["content"] = "\n".join(content_lines).strip()
            entries.append(entry)
        else:
            i += 1

    with jsonlines.open("converted_doc.jsonl", mode="w") as writer:
        for entry in entries:
            writer.write(entry)