# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install minimal OS dependencies for healthchecks & compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and install incidentiq package in editable mode
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy telemetry data, demo application, benchmark suite, and assets
COPY data/ ./data/
COPY demo_app.py benchmark.py ./
COPY assets/ ./assets/

# Expose Streamlit port
EXPOSE 8501

# Container healthcheck ensuring Streamlit server is active
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: launch the interactive SRE console
CMD ["streamlit", "run", "demo_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
