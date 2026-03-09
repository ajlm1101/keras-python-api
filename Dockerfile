FROM python:3.12.10

WORKDIR /app

# evitar cache y archivos pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# copiamos requirements, modelo y codigo fuente
COPY requirements.txt .
COPY mobilenetV2_flowers.keras .
COPY main.py .

# instalar dependencias python
RUN pip install --no-cache-dir -r requirements.txt

# exponer puerto de FastAPI
EXPOSE 8000

# ejecutar servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]