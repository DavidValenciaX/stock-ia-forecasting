# ---- Build stage ----
FROM python:3.13-slim-bullseye AS build

# Instala uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copia el código fuente y archivos de configuración
WORKDIR /app
COPY . /app

# Instala las dependencias necesarias
RUN apt-get update && apt-get install -y --no-install-recommends libc-dev gcc g++ cmake git musl-dev make bash build-essential

# Instala las dependencias en el entorno virtual
RUN uv sync --frozen --no-cache

# ---- Runtime stage ----
FROM python:3.13-slim-bullseye

# Instala las dependencias de runtime
RUN apt-get update && apt-get install -y --no-install-recommends

# Copia uv y el entorno virtual desde la etapa de build
COPY --from=build /bin/uv /bin/uvx /bin/
COPY --from=build /app /app

WORKDIR /app

# Expone el puerto interno 8080
EXPOSE 8080

# Copia el archivo .env para variables de entorno
ENV PYTHONUNBUFFERED=1

# Comando de arranque: usa el puerto interno 8080, el externo se mapea con -p $PORT:8080
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8080", "--host", "0.0.0.0"]
