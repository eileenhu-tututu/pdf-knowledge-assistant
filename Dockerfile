FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001
EXPOSE 8501

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8001 & streamlit run app.py --server.address 0.0.0.0 --server.port 8501"]