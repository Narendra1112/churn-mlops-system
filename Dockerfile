FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Render will inject $PORT. We'll use it for Streamlit.
ENV PORT=10000

# Start FastAPI on 8000 (internal), Streamlit on $PORT (public)
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit/app.py --server.port $PORT --server.address 0.0.0.0"]
