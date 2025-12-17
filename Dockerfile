FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy persisted Chroma DB (knowledge base)
COPY .chroma ./.chroma

# Copy app source
COPY . .

EXPOSE 8501

# Streamlit must listen on all interfaces in a container
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
