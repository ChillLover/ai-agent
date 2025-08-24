# FROM python:3.10-bullseye

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     tesseract-ocr \
#     tesseract-ocr-rus \
#     poppler-utils \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# RUN python -m ensurepip --upgrade && \
#     python -m pip install --force-reinstall "pip<25.0" && \
#     pip install --upgrade setuptools wheel

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY ui.py .

# CMD ["python", "ui.py"]


FROM python:3.11-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     tesseract-ocr \
#     tesseract-ocr-rus \
#     poppler-utils \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

RUN python -m pip install --upgrade "pip==23.3.2"

COPY requirements.txt .

# RUN pip install --no-cache-dir gradio==5.43.1 sniffio==1.3.1

RUN pip install --no-cache-dir -r requirements.txt

COPY ui.py .

CMD ["python", "ui.py"]
