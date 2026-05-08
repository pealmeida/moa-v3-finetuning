FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Core handlers
COPY handler.py .
COPY handler_v31.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# LLMFit tools
COPY llmfit/ ./llmfit/

CMD ["python", "-u", "handler_v31.py"]
