FROM python:3.12-slim

WORKDIR /app

RUN python -m ensurepip --upgrade && \
    python -m pip install --force-reinstall "pip<24.0" && \
    pip install --upgrade setuptools wheel

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

COPY .env .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]