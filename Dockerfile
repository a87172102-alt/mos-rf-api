# Etapa 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Etapa 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Usuario no root por seguridad
RUN adduser --disabled-password --gecos "" appuser

# Copiar dependencias instaladas
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Copiar código y modelos
COPY app/ ./app/
COPY models/ ./models/

ENV PATH=/root/.local/bin:$PATH
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
