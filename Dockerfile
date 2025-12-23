FROM python:3.11-slim

WORKDIR /app

# System dependencies for building packages and database connectivity
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    netcat-openbsd \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Copy wait-for-db script for database health checks
COPY wait-for-db.sh /usr/local/bin/wait-for-db
RUN chmod +x /usr/local/bin/wait-for-db

# Expose the service port
EXPOSE 8001


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Wait for database, then start the service
#ENTRYPOINT ["/usr/local/bin/wait-for-db"]"timescaledb:5432",
CMD [ "python3.11", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info", "--access-log"]
