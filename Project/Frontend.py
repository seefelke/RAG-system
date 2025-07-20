import gradio as gr
import random
import webbrowser
import time
import config
import Chatting
import json
import Evaluation
import Extraction
from datasets import Dataset
from langchain_core.messages import AIMessage, HumanMessage

# TODO: Replace with actual backend call
chat_history = []
with open("questions_and_answers.json", "r", encoding="utf-8") as f:
    questions_and_answers = json.load(f)

question_dict = {}

for category in ["einfache_fragen", "schwere_fragen"]:
    for pair in questions_and_answers.get(category, []):
        question = pair["frage"]
        answer = pair["antwort"]
        question_dict[question] = answer

question_list = list(question_dict.keys())

def query_rag_backend(user_input, reference_input, settings):
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
def handle_chat(user_input, reference_input, chat, chunk_size, chunk_overlap, model):
    settings = {"Chunk Size": chunk_size, "Chunk Overlap": chunk_overlap, "Model": model}
    response, scores = query_rag_backend(user_input, reference_input, settings)
    chat.append((user_input, response))
    return chat, gr.update(value=format_scores(scores)), gr.update(value=""), gr.update(value="")

def format_scores(scores):
    return "\n".join([f"{k}: {v}" for k, v in scores.items()])


def on_model_change(selected_model):
    # TODO: backend restart
    global chat_history
    chat_history = []
    return

def on_vectorbase_change():
    Chatting.setup_vectorbase()


def fill_inputs_from_selection(selected_question):
    if selected_question in question_dict:
        return selected_question, question_dict[selected_question]
    else:
        return "", ""

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
            gr.Markdown("### 📊 Evaluation Scores")
            score_box = gr.Textbox(label="Evaluation", lines=6, interactive=False)
    question_selector.change(
        fn=fill_inputs_from_selection,
        inputs=[question_selector],
        outputs=[user_input, reference_input]
    )
    model.change(fn=on_model_change, inputs = [model])
    chat_state = gr.State([])

    send_button.click(
        handle_chat,
        inputs=[user_input, reference_input, chat_state, chunk_size, chunk_overlap, model],
        outputs=[chatbot, score_box, user_input, reference_input],
    )
    user_input.submit(
        handle_chat,
        inputs=[user_input, reference_input, chat_state, chunk_size, chunk_overlap, model],
        outputs=[chatbot, score_box, user_input, reference_input],
    )

demo.launch(prevent_thread_lock=True)

#time.sleep(5)
webbrowser.open("http://127.0.0.1:7860")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")