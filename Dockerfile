# ---- Python backend image for open-agent ----
FROM python:3.12-slim AS backend

WORKDIR /app

# System deps for FAISS and document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[all,dev]"

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY examples/ ./examples/

# Expose API port
EXPOSE 8000

# Default: start the API server
CMD ["open-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
