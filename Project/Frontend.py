import gradio as gr
import webbrowser
import time
import config
import Chatting
import json
import Evaluation
import Extraction
from datasets import Dataset
import matplotlib.pyplot as plt
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []
reload_vectorbase = False
reload_model = False
with open("questions_and_answers.json", "r", encoding="utf-8") as f:
    questions_and_answers = json.load(f)

question_dict = {}
score_history = []

for category in ["einfache_fragen", "schwere_fragen"]:
    for pair in questions_and_answers.get(category, []):
        question = pair["frage"]
        answer = pair["antwort"]
        question_dict[question] = answer

question_list = list(question_dict.keys())


def query_rag_backend(user_input, reference_input):
    infer_time = time.time()
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
    infer_time = time.time() - infer_time
    eval_time = time.time()
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
        "F1": round(max(0, f1.item()), 3),
        "Recall": round(max(0, recall.item()), 3),
        "Precision": round(max(0, precision.item()), 3),
        "Answer Accuracy": round(ragas_result["nv_accuracy"][0], 3),
        "Faithfulness": round(ragas_result["faithfulness"][0], 3),
        "Context Precision": round(ragas_result["context_precision"][0], 3)
    }
    eval_time = time.time() - eval_time
    return result["answer"], evaluation_scores, round(infer_time, 3), round(eval_time, 3), ragas_dataset["contexts"]


# handle chatbot + evaluation
def handle_chat(user_input, reference_input, chat):
    response, scores, infer_time, eval_time, contexts = query_rag_backend(user_input, reference_input)
    chat.append((user_input, response))
    score_history.append(scores.copy())
    scores["Inference Time"] = infer_time
    scores["Evaluation Time"] = eval_time
    return chat, gr.update(value=format_scores(scores)), gr.update(value=""), gr.update(value=""), gr.update(
        value=contexts)


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


def update_plots():
    last_run = score_history[-1]
    metrics = list(last_run.keys())
    values = [last_run[metric] for metric in metrics]
    box, ax = plt.subplots()
    ax.bar(metrics, values, color='skyblue')
    ax.set_title("Evaluation Metrics - Latest Run")
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1.0, 2.0])
    ax.set_ylabel("Score")
    ax.tick_params(axis='x', labelsize=5)
    box.tight_layout()
    data = {metric: [entry[metric] for entry in score_history[-10:]] for metric in metrics}
    history, ax2 = plt.subplots()
    for metric, values in data.items():
        x = range(1, len(values) + 1)
        ax2.plot(x, values, marker='o', markersize=3, label=metric)
    ax2.set_title("Evaluation Metrics - Last 10")
    ax2.set_xlabel('Run Number (last 10)')
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 2)
    ax2.set_xticks(range(1, len(values) + 1))
    ax2.set_yticks([0, 0.5, 1.0, 2.0])
    history.legend()
    history.tight_layout()
    return box, history


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
            score_box = gr.Textbox(label="Evaluation", lines=8, interactive=False)
    with gr.Row():
        boxplot = gr.Plot()
        history_plot = gr.Plot()
    with gr.Row():
        context_box = gr.Textbox(label="Retrieved Context", interactive=False)
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
        outputs=[chatbot, score_box, user_input, reference_input, context_box]
    ).then(
        update_plots,
        inputs=None,
        outputs=[boxplot, history_plot]
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
