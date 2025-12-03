# Dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential \
        netcat-openbsd \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy project
COPY . /app

# Ensure src/.env exists with (container-friendly) defaults
# We only create it if it doesn't exist already.
RUN mkdir -p src && \
    if [ ! -f src/.env ]; then \
      cat > src/.env << 'EOF'
API_BASE_IP=localhost
API_BASE_PORT=27099
DB_USER=User
DB_PASSWORD=Pass
DB_IP=mongo
DB_PORT=27017
DB_NAME=Steam_Project
EOF \
    ; fi

# Create venv with uv and install deps (build-time, so runtime is instant)
RUN uv venv .venv -p 3.13.7 && \
    uv pip install -r requirements.txt

EXPOSE 8501
EXPOSE 27099

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
