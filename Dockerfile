
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit/ ./.streamlit/
COPY dashboard/ ./dashboard/
COPY data/dashboard/ ./data/dashboard/
COPY models/random_forest_model.joblib ./models/random_forest_model.joblib

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
