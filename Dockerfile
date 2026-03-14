# Imagen base
FROM python:3.12.10

# Directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copiamos archivos del host al contenedor
COPY requirements.txt .
COPY mobilenetV2_flowers.keras .
COPY main.py .

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos puerto de la API
EXPOSE 8000

# Iniciamos la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]