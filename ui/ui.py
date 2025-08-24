import gradio as gr
import requests
import json
from pdf2image import convert_from_path
from pytesseract import image_to_string


def get_preds(request):
    images = convert_from_path(request)
    
    text = ""
    for image in images:
       page_text = image_to_string(image, lang="rus+eng")
       text += f"{page_text}"
    
    text = text.replace("\n", "")

    preds = requests.post("http://api-llm:8000/check_request", json={"request": text})

    if preds.status_code == 200:
        preds = json.loads(preds.text)["Answer"]
        preds = "\n".join(str(value) for value in preds.values())
   
        return preds
    
    else:
        return preds.status_code


with gr.Blocks() as demo:
    with gr.Column():
      input = gr.File(file_types=[".pdf"])
      sub_button = gr.Button(value="Получить решение по запросу")
      output = gr.Textbox(placeholder="Решение по вашему обращению")
      sub_button.click(get_preds, input, output)


demo.launch(server_name="0.0.0.0")
