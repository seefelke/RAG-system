from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain.memory import ConversationBufferWindowMemory
from config import *
import Extraction
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAI

retriever = None

history_aware_retriever = None
question_answer_chain = None
retrieval_chain_history = None
retrieval_chain = None

llm = None


# Current system
# System with history reformulates the current user question based on chat history
# (query, conversation history) -> LLM -> rephrased query -> retriever -> LLM
def setup_chatbot():
    global llm
    use_openai = USE_OPENAI
    chat_model = MODEL_NAME
    if use_openai:
        llm = OpenAI(
            model_name="gpt-4o-mini-2024-07-18",
            temperature=0,
            openai_api_key=os.environ.get('OPENAI_API_KEY')
        )
    else:
        llm = ChatOllama(model=chat_model)
    HISTORY_TEMPLATE = ChatPromptTemplate(
        [
            ("system", HISTORY_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    PROMPT_TEMPLATE_NO_HISTORY = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, HISTORY_TEMPLATE
    )
    question_answer_chain = create_stuff_documents_chain(llm, PROMPT_TEMPLATE)

    global retrieval_chain_history
    retrieval_chain_history = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    global retrieval_chain
    retrieval_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, PROMPT_TEMPLATE_NO_HISTORY))


def setup_vectorbase():
    global retriever
    retriever = Extraction.get_vectorstore().as_retriever(search_type="mmr", search_kwargs={"k": CHUNK_AMOUNT, "lambda_mult": 0.25})


def continuous_chatting(rag_chain):
    while True:
        chat_history = []
        query = input("\nStelle eine Frage (oder 'exit'): ")
        if query.lower() == "exit":
            break
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        #result = rag_chain.invoke({"question": query,
        #                          "chat_history": memory.chat_memory.messages})
        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=result["answer"]),
            ])
        print("\nAntwort:", result["answer"])
        #print("\nSource:", result["source_documents"])


setup_vectorbase()
setup_chatbot()

# Old system

#prompt_template = PromptTemplate(
#    template=(
#        "Du bist ein Helfer um Fragen in einem Museum zu beantworten. "
#        "Antworte im Dialog kurz und präzise in gesprochener Sprache und versuche dich auf wenige Sätze zu "
#        "beschränken."
#        "Antworte immer nur auf deutsch"
#        "Nutze den folgenden Kontext zur Museums Ausstellung um die Fragen zu beantworten.\n\n"
#        "Kontext:\n{context}\n\n"
#        "Verlauf:\n{chat_history}\n\n"
#        "Frage:\n{question}\n\n"
#    ),
#    input_variables=["chat_history", "context", "question"]
#)

#memory = ConversationBufferWindowMemory(
#    k=5,  # Number of conversation turns (or messages) to keep
#    memory_key="chat_history",
#    #return_messages=True,
#    output_key="answer"
#)
#qa_chain = ConversationalRetrievalChain.from_llm(
#    llm=llm,
#    retriever=retriever,
#    memory=memory,
#    combine_docs_chain_kwargs={"prompt": prompt_template},
#    return_source_documents=False,
#    output_key="answer"
#)
