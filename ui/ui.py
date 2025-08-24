# import gradio as gr
# import requests
# import json
# from pdf2image import convert_from_path
# from pytesseract import image_to_string


# def get_preds(request):
#     # images = convert_from_path(request)
    
#     # text = ""
#     # for image in images:
#     #    page_text = image_to_string(image, lang="rus+eng")
#     #    text += f"{page_text}"
    
#     # text = text.replace("\n", "")

#     preds = requests.post("http://api-llm:8000/check_request", json={"request": request})

#     if preds.status_code == 200:
#         preds = json.loads(preds.text)["Answer"]
#         preds = "\n".join(str(value) for value in preds.values())
   
#         return preds
    
#     else:
#         return preds.status_code


# with gr.Blocks() as demo:
#     with gr.Column():
#       input = gr.File(file_types=[".pdf"])
#       sub_button = gr.Button(value="Получить решение по запросу")
#       output = gr.Textbox(placeholder="Решение по вашему обращению")
#       sub_button.click(get_preds, input, output)


# demo.launch(server_name="0.0.0.0")

import pandas as pd
import requests
import tempfile
import gradio as gr


def get_preds(data):
    data = data.name
    data = pd.read_csv(data)
    data = data.to_dict(orient="records")

    response = requests.post("http://api-obesity:3000/predict_features", json={"data": data}) # api:8000 | "https://obesity.projects-pea.ru/api/predict_features" | "http://api-obesity:3000/predict_features"

    if response.status_code == 200:
        preds = pd.read_json(response.json()["Answer"], orient="records")
        preds.rename(columns={0: "Predictions"}, inplace=True)
        result = pd.concat([preds, pd.DataFrame.from_dict(data, orient="columns")], axis=1)

        result["Predictions"] = result["Predictions"].map({
                0: "Insufficient_Weight",
                1: "Normal_Weight",
                2: "Overweight_Level_I",
                3: "Overweight_Level_II",
                4: "Obesity_Type_I",
                5: "Obesity_Type_II",
                6: "Obesity_Type_III"
            })
    
        temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        result.to_csv(temp_file.name, index=False)

        return result, temp_file.name                                                     


with gr.Blocks() as demo:
    input_file = gr.File(label="Загрузить файл", file_types=[".csv"])
    process_button = gr.Button("Обработать файл")
    output = gr.DataFrame(label="Полученные предсказания")
    download_button = gr.DownloadButton(label="Скачать предсказания")

    process_button.click(
        fn=get_preds,
        inputs=input_file,
        outputs=[output, download_button],
    )

demo.launch(server_name="0.0.0.0")

