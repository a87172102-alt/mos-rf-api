# Etapa 1: Build (como root)
FROM public.ecr.aws/docker/library/python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
# Instalar en el directorio global del sistema, no en /root/.local
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Etapa 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Crear usuario no root
RUN adduser --disabled-password --gecos "" appuser

# Copiar dependencias a un directorio accesible por appuser
COPY --from=builder /app/deps /app/deps

# Copiar código y modelos (con propiedad para appuser)
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ ./models/

# Configurar PATH para que uvicorn sea encontrado desde /app/deps/bin
ENV PATH=/app/deps/bin:$PATH
ENV PYTHONPATH=/app/deps

# Cambiar a usuario no root AL FINAL
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
