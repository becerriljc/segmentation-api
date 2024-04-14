import numpy as np
import io
import PIL
from tensorflow import keras
from tensorflow.keras.utils import load_img
from PIL import Image
import tensorflow as tf
from fastapi.responses import StreamingResponse
import os

model_path = 'src/api/v1/models/model.keras'
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((160, 160))  # Resize to the input size expected by your model
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0  # Normalize if your model expects this
    return image_array

def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256
    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]
    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out
    
def create_image_from_mask(prediction):
    try:
        """Quick utility to display a model's prediction."""
        mask = np.argmax(prediction, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        return img

    except Exception as e:
        print('create_image_from_mask :: error: {}'.format(e))

        raise Exception('Error al crear la imagen mascara')

async def get_image_filtered(file):
    try:
        image_data = await file.read()
        image_array = preprocess_image(io.BytesIO(image_data))
        prediction = model.predict(image_array)
        mask_image = create_image_from_mask(prediction[0])  # Assuming batch size 1

        print('INFO :: get_image_filtered :: mask_image: {}'.format(mask_image))

        # Save the PIL image to a BytesIO object and return it as a StreamingResponse
        img_io = io.BytesIO()
        mask_image.save(img_io, 'JPEG')
        img_io.seek(0)
        return StreamingResponse(img_io, media_type='image/jpeg')
    except Exception as e:
        print('get_image_filtered :: error: {}'.format(e))

        raise Exception('Error al procesar la imagen')