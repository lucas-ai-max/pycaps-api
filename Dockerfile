FROM python:3.11-slim

# Instalar FFmpeg e dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    chromium \
    chromium-driver \
    fonts-liberation \
    fonts-noto \
    libglib2.0-0 \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libegl1 \
    libgles2 \
    libgl1 \
    libxkbcommon0 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar PyTorch CPU-only (evita baixar 5GB de CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar pycaps + whisper + API
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    "pycaps @ git+https://github.com/francozanardi/pycaps.git" \
    openai-whisper \
    playwright

# Instalar browser do Playwright para renderização CSS
RUN python -m playwright install chromium --with-deps || true

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV CHROMIUM_PATH=/usr/bin/chromium

# Criar diretório de trabalho para vídeos temporários
RUN mkdir -p /tmp/pycaps-work && chmod 777 /tmp/pycaps-work

# Copiar código da API
COPY app.py /app/app.py

EXPOSE 8000

# Health check para o Easypanel
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]
