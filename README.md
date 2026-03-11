# Keras API

## Descripción general

Este proyecto implementa una **API REST para clasificación de imágenes de flores** utilizando un modelo de **Deep Learning basado en MobileNetV2** entrenado con TensorFlow/Keras.

La API está desarrollada con **FastAPI** y permite enviar una imagen mediante una petición HTTP y recibir como respuesta la **clase de flor predicha junto con su nivel de confianza**.

El modelo cargado (`mobilenetV2_flowers.keras`) ha sido entrenado para clasificar imágenes en cinco categorías de flores:

* Dandelion
* Daisy
* Tulips
* Sunflowers
* Roses

La API recibe una imagen, la procesa para adaptarla al formato requerido por el modelo y devuelve la predicción correspondiente.

---

# Arquitectura del sistema

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

# Librerías utilizadas

* **TensorFlow**: framework de Deep Learning utilizado para cargar el modelo y realizar predicciones.
* **NumPy**: manipulación de arrays y cálculo de la clase con mayor probabilidad.
* **Pillow (PIL)**: apertura, conversión y redimensionamiento de imágenes.
* **FastAPI**: framework para crear la API REST.
* **logging**: registro de eventos y errores de la aplicación.
* **io**: manejo de datos binarios de la imagen recibida.

Además se utiliza:

`tensorflow.keras.applications.mobilenet_v2.preprocess_input`

para aplicar el preprocesamiento necesario para el modelo MobileNetV2.

---

# Ejecución del proyecto

Este proyecto requiere **Python 3.12.10**. Una vez clonado el repositorio, se recomienda el uso de PyCharm o Visual Studio Code.

## Instalación de dependencias

Las dependencias necesarias se encuentran en el archivo `requirements.txt`.

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar la API

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
# Explicación del código

## Configuración del modelo

El código define algunas constantes principales:

```python
CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]
MODEL_PATH = "mobilenetV2_flowers.keras"
IMG_SIZE = (160, 160)
```

**CLASS_NAMES:** Lista de clases que el modelo puede predecir.

**MODEL_PATH:** Ruta del archivo del modelo entrenado en formato `.keras`.

**IMG_SIZE:** Tamaño al que se redimensionan todas las imágenes antes de enviarlas al modelo.

---

## Carga del modelo

Durante el arranque de la aplicación, el modelo se carga en memoria:

```python
model = tf.keras.models.load_model(MODEL_PATH)
```

Esto permite que las predicciones posteriores sean rápidas, evitando recargar el modelo en cada petición.

## Funciones del código

### preprocess_image()

```python
def preprocess_image(image_bytes):
```

Esta función se encarga de **preparar la imagen recibida para ser utilizada por el modelo de Deep Learning**.

#### Parámetros

* `image_bytes`: imagen recibida en formato binario.

#### Proceso que realiza

1. Abre la imagen desde los bytes recibidos.
2. Convierte la imagen a formato RGB.
3. Redimensiona la imagen a `160x160`.
4. Convierte la imagen en un array de NumPy.
5. Aplica el preprocesado requerido por MobileNetV2.
6. Añade una dimensión adicional para representar el batch.

#### Salida

Devuelve un tensor con forma:

```
(1, 160, 160, 3)
```

Este formato es el esperado por el modelo para realizar predicciones.

## API REST

La aplicación se crea utilizando FastAPI:

```python
app = FastAPI()
```

## Endpoint de predicción

### POST `/predict`

Este endpoint recibe una imagen y devuelve la predicción del modelo.

#### Parámetros

* `file`: archivo de imagen enviado mediante `multipart/form-data`.

#### Proceso interno

1. Lee el archivo subido.
2. Preprocesa la imagen.
3. Ejecuta la predicción con el modelo.
4. Obtiene:

   * índice de la clase más probable
   * nombre de la clase
   * nivel de confianza

#### Código principal

```python
predictions = model.predict(processed_image)
predicted_index = np.argmax(predictions)
predicted_class = CLASS_NAMES[predicted_index]
confidence = np.max(predictions)
```

#### Respuesta

La API devuelve un JSON con la siguiente estructura:

```json
{
  "filename": "flower.jpg",
  "predicted_class": "sunflowers",
  "confidence": 0.97
}
```

#### Manejo de errores

Si ocurre algún problema durante la predicción, se captura la excepción y se devuelve un error HTTP:

```python
HTTPException(status_code=500)
```

Además, el error se registra en el sistema de logs para facilitar la depuración.

---

# Dockerización y despliegue

La aplicación puede ejecutarse dentro de un **contenedor Docker** para facilitar su despliegue y garantizar que se ejecute siempre en el mismo entorno.

## Dockerfile

El proyecto incluye un fichero `Dockerfile` para construir la imagen de la API. Este archivo:

* Utiliza **Python 3.12.10** como imagen base.
* Copia el código fuente, el modelo y las dependencias al contenedor.
* Instala las librerías definidas en `requirements.txt`.
* Expone el puerto **8000** para acceder a la API.
* Ejecuta el servidor **Uvicorn** que lanza la aplicación FastAPI.

## Construcción de la imagen

Para construir la imagen Docker:

```bash
docker build -t keras-api .
```

## Ejecución del contenedor

Para iniciar el contenedor:

```bash
docker run -p 8000:8000 keras-api
```

Esto expone la API en el puerto **8000** del host.

## Acceso a la API

Una vez iniciado el contenedor, la API estará disponible en:

```
http://localhost:8000
```

Documentación interactiva de FastAPI:

```
http://localhost:8000/docs
```

---

# Posibles mejoras

* Añadir validación de tipo de imagen.
* Implementar predicción por lotes.
* Mejorar la seguridad del contenedor mediante grupos y usuarios.
