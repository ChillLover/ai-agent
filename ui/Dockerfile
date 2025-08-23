FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m ensurepip --upgrade && \
    python -m pip install --force-reinstall "pip<24.0" && \
    pip install --upgrade setuptools wheel

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ui.py .

CMD ["python", "ui.py"]
