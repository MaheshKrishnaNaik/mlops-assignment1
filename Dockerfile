FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir pandas numpy "scikit-learn>=1.5,<2" joblib fastapi uvicorn pydantic prometheus-client

COPY src/ src/
COPY models/model.joblib models/model.joblib

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
