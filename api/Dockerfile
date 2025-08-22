FROM python:3.11-slim

WORKDIR /app

RUN pip uninstall -y pip && apt-get update && apt-get install -y python3-pip --reinstall

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]