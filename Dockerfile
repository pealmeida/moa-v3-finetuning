FROM runpod/serverless-sdk:latest
RUN pip install --no-cache-dir scipy numpy datasets
COPY handler.py /
