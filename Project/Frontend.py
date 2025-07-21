import gradio as gr
import webbrowser
import time
import config
import Chatting
import json
import Evaluation
import Extraction
from datasets import Dataset
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []
reload_vectorbase = False
reload_model = False
with open("questions_and_answers.json", "r", encoding="utf-8") as f:
    questions_and_answers = json.load(f)

question_dict = {}

for category in ["einfache_fragen", "schwere_fragen"]:
    for pair in questions_and_answers.get(category, []):
        question = pair["frage"]
        answer = pair["antwort"]
        question_dict[question] = answer

question_list = list(question_dict.keys())


def query_rag_backend(user_input, reference_input):
    llm = Chatting.llm
    if config.USE_HISTORY:
        result = Chatting.retrieval_chain_history.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        chat_history.extend(
            [
                HumanMessage(content=user_input),
                AIMessage(content=result["answer"]),
            ])
    else:
        result = Chatting.retrieval_chain.invoke({
            "input": user_input
        })

    prediction = result.get('answer', '') if isinstance(result, dict) else str(result)
    sources = result.get('context', [])
    ragas_dataset = []
    ragas_dataset.append({
        "question": user_input,
        "ground_truth": reference_input,
        "answer": prediction,
        "contexts": [doc.page_content for doc in sources]
    })
    ragas_dataset = Dataset.from_list(ragas_dataset)
    bertscore_result = Evaluation.evaluate_bertscore([prediction], [reference_input])
    ragas_result = Evaluation.evaluate_ragas(llm, ragas_dataset, Extraction.embedding)

    precision = bertscore_result["precision"]
    recall = bertscore_result["recall"]
    f1 = bertscore_result["f1"]
    evaluation_scores = {
        "F1": f1.item(),
        "Recall": recall.item(),
        "Precision": precision.item(),
        "Answer Accuracy": ragas_result["nv_accuracy"],
        "Faithfulness": ragas_result["faithfulness"],
        "Context Precision": ragas_result["context_precision"]
    }

    return result["answer"], evaluation_scores


# handle chatbot + evaluation
def handle_chat(user_input, reference_input, chat):
    response, scores = query_rag_backend(user_input, reference_input)
    chat.append((user_input, response))
    return chat, gr.update(value=format_scores(scores)), gr.update(value=""), gr.update(value="")


def format_scores(scores):
    return "\n".join([f"{k}: {v}" for k, v in scores.items()])


def on_model_change():
    global reload_model
    reload_model = True


def on_vectorbase_change():
    global reload_vectorbase
    reload_vectorbase = True


def fill_inputs_from_selection(selected_question):
    if selected_question in question_dict:
        return selected_question, question_dict[selected_question]
    else:
        return "", ""

def apply_settings(chunk_size, chunk_overlap, model, vectorstore, progress=gr.Progress()):
    global chat_history, reload_vectorbase, reload_model
    progress(0, desc="Applying settings ...")
    config.CHUNK_SIZE = chunk_size
    config.CHUNK_OVERLAP = chunk_overlap
    config.MODEL_NAME = model
    if model == config.MODELS[1]:
        config.USE_OPENAI = True
    else:
        config.USE_OPENAI = False
    if vectorstore == config.VECTORSTORES[0]:
        config.USE_FAISS = True
    else:
        config.USE_FAISS = False
    config.STORE_TYPE = vectorstore
    progress(0.2, desc="Applying settings ...")
    if reload_vectorbase:
        reload_vectorbase = False
        Chatting.setup_vectorbase()
    progress(0.5, desc="Applying settings ...")
    if reload_model:
        chat_history = []
        reload_model = False
        Chatting.setup_chatbot()
    progress(1, desc="Settings applied!")
    time.sleep(0.2)
    print("Settings applied")
    return gr.update("Status:")


# build UI
with gr.Blocks(title="RAG Assistant with Evaluation") as demo:
    gr.Markdown("## RAG Assistant with Evaluation")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat with RAG Bot")
            question_selector = gr.Dropdown(
                label="Select a premade question-answer pair",
                choices=question_list,
                interactive=True,
                allow_custom_value=False
            )
            user_input = gr.Textbox(placeholder="Type your question...", label="Your input")
            reference_input = gr.Textbox(
                placeholder="Enter expected/reference answer (optional)", label="Reference Answer")
            send_button = gr.Button("Send")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            chunk_size = gr.Number(value=config.CHUNK_SIZE, label="Chunk Size")
            chunk_overlap = gr.Number(value=config.CHUNK_OVERLAP, label="Chunk Overlap")
            model = gr.Dropdown(label="Model", choices=config.MODELS, value=config.MODELS[0], interactive=True)
            vectorstore = gr.Dropdown(label="Vectorstore", choices=config.VECTORSTORES,
                                      value=config.VECTORSTORES[0] if config.USE_FAISS else config.VECTORSTORES[1],
                                      interactive=True)
            apply_button = gr.Button("Apply Settings")
            apply_status = gr.Markdown("Status:")
            gr.Markdown("### 📊 Evaluation Scores")
            score_box = gr.Textbox(label="Evaluation", lines=6, interactive=False)
    question_selector.change(
        fn=fill_inputs_from_selection,
        inputs=[question_selector],
        outputs=[user_input, reference_input]
    )
    model.change(fn=on_model_change, inputs=[])
    vectorstore.change(fn=on_vectorbase_change, inputs=[])
    chunk_size.change(fn=on_vectorbase_change, inputs=[])
    chunk_overlap.change(fn=on_vectorbase_change, inputs=[])
    chat_state = gr.State([])
    buttons = [send_button, apply_button]


    send_button.click(
        fn=lambda: [gr.update(interactive=False)] * len(buttons),
        outputs=buttons
    ).then(
        handle_chat,
        inputs=[user_input, reference_input, chat_state],
        outputs=[chatbot, score_box, user_input, reference_input],
    ).then(
        fn=lambda: [gr.update(interactive=True)] * len(buttons),
        outputs=buttons
    )
    user_input.submit(
        handle_chat,
        inputs=[user_input, reference_input, chat_state],
        outputs=[chatbot, score_box, user_input, reference_input],
    )
    apply_button.click(
        fn=lambda: [gr.update(interactive=False)] * len(buttons),
        outputs=buttons
    ).then(
        apply_settings,
        inputs=[chunk_size, chunk_overlap, model, vectorstore],
        outputs=[apply_status]
    ).then(
        fn=lambda: [gr.update(interactive=True)] * len(buttons),
        outputs=buttons
    )

demo.launch(prevent_thread_lock=True)

webbrowser.open("http://127.0.0.1:7860")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
