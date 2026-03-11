import io
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(level = logging.INFO)
log = logging.getLogger("Keras_API")
log.info("Iniciando Keras API...")

CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]
MODEL_PATH = "mobilenetV2_flowers.keras"
IMG_SIZE = (160, 160)

log.info("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
log.info("Modelo cargado con exito!")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

app = FastAPI()

@app.post("/predict")
async def predict_img(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        predictions = model.predict(processed_image, verbose=0)
        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions))
        log.info("Recibida peticion de prediccion, devolviendo resultados...")
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        log.error("Se ha producido una excepción al predecir: %s", e)
        raise HTTPException(status_code=500, detail="Se ha producido una excepcion al predecir")