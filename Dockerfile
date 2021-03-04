FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api /api
COPY model.joblib /model.joblib

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
