FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip supertools

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ui.py .

CMD ["python", "ui.py"]
