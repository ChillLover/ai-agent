# FROM python:3.10-bullseye

# WORKDIR /app

# RUN python -m ensurepip --upgrade && \
#     python -m pip install --force-reinstall "pip<24.0" && \
#     pip install --upgrade setuptools wheel

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY api.py .

# COPY .env .

# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10-bullseye

WORKDIR /app

RUN python -m pip install --upgrade "pip==23.3.2"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY .env .

COPY api.py .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]