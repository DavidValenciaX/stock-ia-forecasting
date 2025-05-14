# ---- Build stage ----
FROM python:3.13-alpine AS build

# Instala uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copia el código fuente y archivos de configuración
WORKDIR /app
COPY . /app

# Instala las dependencias necesarias para compilar psycopg2
RUN apk add --no-cache postgresql-dev gcc musl-dev

# Instala las dependencias en el entorno virtual
RUN uv sync --frozen --no-cache

# ---- Runtime stage ----
FROM python:3.13-alpine

# Instala las dependencias de runtime para PostgreSQL
RUN apk add --no-cache postgresql-libs

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
