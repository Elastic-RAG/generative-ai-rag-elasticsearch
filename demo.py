import gradio as gr
from app import main as app


def index_data_source():
    app.index_data_source()
    return "Successfully indexed data source"

def chatbot_response(dropdown_value, services, user_query):
    return app.llama_main(user_query, dropdown_value)


with gr.Blocks() as demo:
    index_button = gr.Button("Index Data")
    index_button.click(index_data_source, outputs=gr.Textbox(label="Output"))
    techniques = gr.Dropdown(choices=["Search", "RAG", "No RAG"], label="Techniques")
    services = gr.Dropdown(choices=["Meta LLama 2-7b"], label="Services")
    user_query = gr.Textbox(label="User Query")
    submit_button = gr.Button("Submit")
    chatbot_output = gr.Textbox(label="Chatbot Response")
    submit_button.click(
        chatbot_response,
        inputs=[techniques, services, user_query],
        outputs=chatbot_output
    )

demo.launch()
