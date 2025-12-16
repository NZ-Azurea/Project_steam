FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS llama_build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        cmake \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && \
    ldconfig

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && \
    cmake --build /opt/llama.cpp/build -j && \
    ln -sf /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server


FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        netcat-openbsd \
        git && \
    rm -rf /var/lib/apt/lists/*

# Copy llama-server + its shared libs from the build stage
COPY --from=llama_build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama_build /opt/llama.cpp/build/bin/*.so* /usr/local/lib/

# Make sure the runtime linker can find them
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama-cpp.conf && ldconfig

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy project files into image
COPY . /app

# Ensure src/.env exists with container-friendly defaults (only if missing)
RUN mkdir -p src \
 && if [ ! -f src/.env ]; then \
      printf '%s\n' \
'API_BASE_IP=localhost' \
'API_BASE_PORT=27099' \
'DB_USER=User' \
'DB_PASSWORD=Pass' \
'DB_IP=mongo' \
'DB_PORT=27017' \
'DB_NAME=Steam_Project' > src/.env; \
    fi

# Create uv venv and install Python dependencies at build time
RUN uv venv .venv -p 3.13.7 && \
    uv pip install -r requirements.txt

# Expose app ports
EXPOSE 8501
EXPOSE 27099
EXPOSE 8080

# Entrypoint script (starts DB init + API + Streamlit)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]