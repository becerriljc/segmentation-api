import numpy as np
import io
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

def create_image_from_mask(prediction):
    try:
        # Convert RGB numpy array to a PIL image
        image = Image.fromarray(prediction.astype('uint8'), 'RGB')

        # Convert RGB image to Grayscale
        gray_image = image.convert('L')

        # Convert grayscale image to numpy array
        gray_array = np.array(gray_image)

        # Apply threshold to create a binary mask
        # You might need to adjust the threshold value based on your specific case
        mask = (gray_array > 128).astype(np.uint8) * 255  # Example threshold of 128

        # Convert the binary mask back to a PIL image
        mask_image = Image.fromarray(mask, mode='L')
        return mask_image
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
