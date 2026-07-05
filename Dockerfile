# ---- Python backend image for open-agent ----
# Multi-stage build: compile wheels in the builder stage, then copy the
# virtual environment into a smaller runtime image.
FROM python:3.12-slim AS builder

WORKDIR /app

# Build dependencies required by FAISS and some native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install dependencies.
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[all]"

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime library required by FAISS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the prepared virtual environment from the builder.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code.
COPY src/ ./src/

# Create a non-root user for runtime security.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Persisted data directories.
ENV OPEN_AGENT_SESSION_STORAGE_DIR=/app/.open_agent_sessions
ENV OPEN_AGENT_OBSERVABILITY_OUTPUT_DIR=/app/.open_agent_traces

# Expose API port.
EXPOSE 8000

# Default: start the API server.
CMD ["open-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
