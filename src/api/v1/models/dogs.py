import numpy as np
import io
from PIL import Image, ImageDraw
import tensorflow as tf
from fastapi.responses import StreamingResponse
import os

model_path = 'src/api/v1/models/model.keras'
model = tf.keras.models.load_model(model_path)

def model_predict(image: Image.Image) -> np.ndarray:
    # Placeholder for your model's prediction logic
    # Let's pretend we're detecting a single object and returning its bounding box
    # Returning a dummy bounding box: [x_min, y_min, x_max, y_max]
    prediction = model.predict(image)
    print(prediction)

    return np.array([50, 50, 200, 200])

def process_image(image: Image.Image, prediction: np.ndarray) -> Image.Image:
    # Use the prediction to modify the image
    # For example, draw a bounding box around a detected object
    draw = ImageDraw.Draw(image)
    draw.rectangle(prediction.tolist(), outline="red", width=3)
    return image

async def get_image_filtered(image):
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    # Process the image - adjust depending on your model's input requirements
    image = image.resize((224, 224))  # Example resize, change to match your model's expected input shape
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)  # Model expects a batch of images

    # Normalize the image data to 0-1 range if your model expects that
    image_array = image_array / 255.0

    # Predict
    # Perform model prediction
    prediction = model_predict(image)

    # Modify the image based on the prediction
    transformed_image = process_image(image, prediction)

    # Convert the PIL image back to bytes to return it
    img_byte_arr = io.BytesIO()
    transformed_image.save(img_byte_arr, format='JPEG')  # Adjust the format as needed
    img_byte_arr.seek(0)  # Go to the start of the BytesIO object

    # Create a StreamingResponse to return the image
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

