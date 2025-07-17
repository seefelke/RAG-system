import gradio as gr
import random
import webbrowser
import time
import config
#import Chatting
#import Evaluation

# TODO: Replace with actual backend call
def query_rag_backend(user_input, settings):
    response = f"Simulated response to: {user_input}"
    evaluation_scores = {
        "F1": round(random.uniform(0.5, 1.0), 2),
        "Accuracy": round(random.uniform(0.5, 1.0), 2),
        "Response Time (ms)": random.randint(150, 500)
    }
    return response, evaluation_scores

# handle chatbot + evaluation
def handle_chat(user_input, chat_history, chunk_size, chunk_overlap, model):
    settings = {"Chunk Size": chunk_size, "Chunk Overlap": chunk_overlap, "Model": model}
    response, scores = query_rag_backend(user_input, settings)
    chat_history.append((user_input, response))
    return chat_history, gr.update(value=format_scores(scores))

def format_scores(scores):
    return "\n".join([f"{k}: {v}" for k, v in scores.items()])


def on_model_change(selected_model):
    # TODO: backend restart
    return

# build UI
with gr.Blocks(title="RAG Assistant with Evaluation") as demo:

    gr.Markdown("## RAG Assistant with Evaluation")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat with RAG Bot")
            user_input = gr.Textbox(placeholder="Type your question...", label="Your input")
            send_button = gr.Button("Send")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            chunk_size = gr.Number(value=150, label="Chunk Size")
            chunk_overlap = gr.Number(value=50, label="Chunk Overlap")
            model = gr.Dropdown(label="Model", choices=config.MODELS, value=config.MODELS[0], interactive=True)
            gr.Markdown("### 📊 Evaluation Scores")
            score_box = gr.Textbox(label="Evaluation", lines=6, interactive=False)
    model.change(fn=on_model_change, inputs = [model])
    chat_state = gr.State([])

    send_button.click(
        handle_chat,
        inputs=[user_input, chat_state, chunk_size, chunk_overlap, model],
        outputs=[chatbot, score_box],
    )
    user_input.submit(
        handle_chat,
        inputs=[user_input, chat_state, chunk_size, chunk_overlap, model],
        outputs=[chatbot, score_box],
    )

demo.launch(prevent_thread_lock=True)

#time.sleep(5)
webbrowser.open("http://127.0.0.1:7860")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")