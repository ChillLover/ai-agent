import gradio as gr
import requests
import json


def get_preds(request):
    preds = requests.post("http://api-ai-llm:8000/check_request", json={"request": request})

    if preds.status_code == 200:
        preds = json.loads(preds.text)["Answer"]
        preds = "\n".join(str(value) for value in preds.values())
   
        return preds
    
    else:
        return preds.status_code


with gr.Blocks() as demo:
    with gr.Column():
      input = gr.Textbox(placeholder="Напишите свое обращение в партию")
      sub_button = gr.Button(value="Получить решение по запросу")
      output = gr.Textbox(placeholder="Решение по вашему обращению")
      sub_button.click(get_preds, input, output)


demo.launch(server_name="0.0.0.0")
