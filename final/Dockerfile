FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration files
COPY pyproject.toml README.md ./

# Copy source code
COPY src ./src

# Install dependencies and the package
RUN pip install --no-cache-dir .

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV RETRIEVAL_MODE=splade
ENV API_MODEL_NAME="qwen/qwen3-32b"

# Expose API port
EXPOSE 8000

# Run the installed package module
CMD ["uvicorn", "contextforge.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
