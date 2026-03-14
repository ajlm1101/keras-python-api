# Keras API

## Descripción general

Este proyecto implementa una API REST para clasificación de imágenes de flores utilizando un modelo entrenado con TensorFlow/Keras.
Recibe una petición HTTP POST `/predict` y recibir como respuesta la clase de flor predicha junto con su nivel de confianza.

El modelo cargado (`mobilenetV2_flowers.keras`) ha sido entrenado para clasificar imágenes en cinco categorías de flores:

* Dandelion
* Daisy
* Tulips
* Sunflowers
* Roses

---

## Arquitectura del sistema

El flujo de ejecución del sistema es el siguiente:

1. Se inicia la aplicación y se carga el modelo de TensorFlow.
2. Un cliente envía una imagen a través del endpoint `/predict`.
3. La imagen se preprocesa para que tenga el formato requerido por MobileNetV2.
4. El modelo realiza la predicción.
5. Se devuelve la clase predicha y la probabilidad asociada.

```
Cliente -> API FastAPI -> Preprocesado imagen -> Modelo MobileNetV2 -> Predicción -> Respuesta JSON
```

---

## Librerías utilizadas

* **TensorFlow**: framework de Deep Learning utilizado para cargar el modelo y realizar predicciones.
* **NumPy**: manipulación de arrays y cálculo de la clase con mayor probabilidad.
* **Pillow (PIL)**: apertura, conversión y redimensionamiento de imágenes.
* **FastAPI**: framework para crear la API REST.
* **logging**: registro de eventos y errores de la aplicación.
* **io**: manejo de datos binarios de la imagen recibida.

Además se utiliza `tensorflow.keras.applications.mobilenet_v2.preprocess_input` para aplicar el preprocesamiento necesario para el modelo MobileNetV2.

---

## Ejecución del proyecto

Este proyecto requiere **Python 3.12.10**.

### Instalación de dependencias

Las dependencias necesarias se encuentran en el archivo `requirements.txt`.

Instalar dependencias:

```bash
pip install -r requirements.txt
```

### Ejecutar la API

Una vez instaladas las dependencias, iniciar el servidor con:

```bash
uvicorn main:app --reload
```

La API estará disponible en:

```
http://localhost:8000
```

Documentación interactiva automática:

```
http://localhost:8000/docs
```

---

## Explicación del código

### Configuración del modelo

El código define algunas constantes principales:

```python
CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]
MODEL_PATH = "mobilenetV2_flowers.keras"
IMG_SIZE = (160, 160)
```

* **CLASS_NAMES:** Lista de clases que el modelo puede predecir.
* **MODEL_PATH:** Ruta del archivo del modelo entrenado en formato `.keras`.
* **IMG_SIZE:** Tamaño al que se redimensionan todas las imágenes antes de enviarlas al modelo.

### Carga del modelo

Durante el arranque de la aplicación, el modelo se carga en memoria:

```python
model = tf.keras.models.load_model(MODEL_PATH)
```

Esto permite que las predicciones posteriores sean rápidas, evitando recargar el modelo en cada petición.

### Preprocesamiento de la imagen

La siguiente funcion se encarga de preparar la imagen recibida para ser utilizada por el modelo:

```python
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
```

Recibe el siguiente parámetro:

* `image_bytes`: imagen recibida en formato binario.

A continuación, se describe el proceso que realiza

1. Abre la imagen desde los bytes recibidos.
2. Convierte la imagen a formato RGB.
3. Redimensiona la imagen a `160x160`.
4. Convierte la imagen en un array de NumPy.
5. Aplica el preprocesado requerido por MobileNetV2.
6. Añade una dimensión adicional para representar el batch.

Devuelve un tensor con forma:

```
(1, 160, 160, 3)
```

Este formato es el esperado por el modelo para realizar predicciones.

### API REST

La aplicación se crea utilizando FastAPI:

```python
app = FastAPI()
```

### Endpoint de predicción

Este endpoint se encarga de recibir la imagen y de devolver la predicción del modelo.

````python
@app.post("/predict")
async def predict_img(file: UploadFile = File(...)):
   try:
      contents = await file.read()
      processed_image = preprocess_image(contents)
      predictions = model.predict(processed_image, verbose=0)
      predicted_index = int(np.argmax(predictions))
      predicted_class = CLASS_NAMES[predicted_index]
      confidence = float(np.max(predictions))
      return {
         "filename": file.filename,
         "predicted_class": predicted_class,
         "confidence": confidence
      }
   except Exception as e:
           raise HTTPException(status_code=500, detail="Se ha producido una excepcion al predecir")
````

 Recibe el siguiente parámetro:

* `file`: archivo de imagen enviado mediante `multipart/form-data`.

Internamente:

1. Lee el archivo subido.
2. Preprocesa la imagen.
3. Ejecuta la predicción con el modelo.
4. Obtiene:

   * Índice de la clase más probable
   * Nombre de la clase
   * Nivel de confianza

La API devuelve un JSON con la siguiente estructura:

```json
{
  "filename": "flower.jpg",
  "predicted_class": "sunflowers",
  "confidence": 0.97
}
```

Si ocurre algún problema durante la predicción, se captura la excepción y se devuelve un error HTTP:

---

## Dockerización y despliegue

La aplicación puede ejecutarse dentro de un **contenedor Docker** para facilitar su despliegue y garantizar que se ejecute siempre en el mismo entorno.

### Dockerfile

El proyecto incluye un fichero `Dockerfile` para construir la imagen de la API. Este archivo:

* Utiliza **Python 3.12.10** como imagen base.
* Copia el código fuente, el modelo y las dependencias al contenedor.
* Instala las librerías definidas en `requirements.txt`.
* Expone el puerto **8000** para acceder a la API.
* Ejecuta el servidor **Uvicorn** que lanza la aplicación FastAPI.

```Dockerfile
FROM python:3.12.10
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
COPY mobilenetV2_flowers.keras .
COPY main.py .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Construcción de la imagen

Para construir la imagen Docker:

```bash
docker build -t keras-api .
```

### Ejecución del contenedor

Para iniciar el contenedor:

```bash
docker run --name keras-api -p 8000:8000 keras-api
```

Esto expone la API en el puerto **8000** del host.

### Acceso a la API

Una vez iniciado el contenedor, la API estará disponible en:

```
http://localhost:8000
```

Documentación interactiva de FastAPI:

```
http://localhost:8000/docs
```

---

## Posibles mejoras

* Añadir validación de tipo de imagen.
* Agregar más datos a la respuesta que pueda proporcionar el modelo.
* Implementar predicción por lotes en caso de recibir varias imágenes.
* Mejorar la seguridad del contenedor mediante grupos y usuarios.
