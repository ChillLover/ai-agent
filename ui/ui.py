import gradio as gr
import requests
import json
import os


def get_preds(pdf_file):
    with open(pdf_file.name, "rb") as file:
        files = {"file": (os.path.basename(pdf_file.name), file, "application/pdf")}
        preds = requests.post("http://api-llm:3000/check_request", files=files) # "http://api-llm:3000/check_request" | "http://api:8000/check_request"

    if preds.status_code == 200:
        preds = json.loads(preds.text)["Answer"]
        preds = "\n".join(str(value) for value in preds.values())
   
        return preds
    
    else:
        return preds.status_code


with gr.Blocks() as demo:
    with gr.Column():
      input = gr.File(file_types=[".pdf"], file_count="single")
      sub_button = gr.Button(value="Получить решение по запросу")
      output = gr.Textbox(placeholder="Решение по вашему обращению")
      sub_button.click(get_preds, input, output)


demo.launch(server_name="0.0.0.0")
