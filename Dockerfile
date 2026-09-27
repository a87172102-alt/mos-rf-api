# Etapa 1: Build
FROM public.ecr.aws/docker/library/python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
# Instalar dependencias en un directorio accesible por el usuario final
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Etapa 2: Runtime
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Crear usuario no root para seguridad
RUN adduser --disabled-password --gecos "" appuser

# Copiar dependencias desde la etapa de build
COPY --from=builder /app/deps /app/deps

# Copiar código y modelos con permisos para appuser
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ ./models/

# Configurar PATH y PYTHONPATH para que uvicorn y los paquetes sean accesibles
ENV PATH=/app/deps/bin:$PATH
ENV PYTHONPATH=/app/deps

# Cambiar a usuario no root (después de todas las instalaciones)
USER appuser

# Exponer el puerto 8000 (no el 80, que requiere root)
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
