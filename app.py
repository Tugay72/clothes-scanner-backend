import threading
from flask import Flask, request, jsonify, send_file # type: ignore
import cv2 # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np # type: ignore
import os
from math import sqrt
from colornamer import get_color_from_rgb # type: ignore
import time
from io import BytesIO
from PIL import Image # type: ignore

base_color_mapping = {
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "blue": "blue",
    "purple": "purple",
    "pink": "pink",
    "brown": "brown",
    "gray": "gray",
    "black": "black",
    "white": "white",
    "khaki": "brown",
    "red violet": "red",
    "olive green": "green",
    "lavender": "purple",
    "dark orange": "orange",
    "steel blue": "blue",
    "slate gray": "gray",
    "peru": "brown",
    "light coral": "pink",
    "misty rose": "pink",
    "chocolate": "brown",
    "lime green": "green",
    "firebrick": "red",
    "medium orchid": "purple",
    "dark sea green": "green",
    "green blue": "blue",
    "violet blue": "purple",
    "yellow green": "yellow",
    "olive": "black",
    "sienna": "brown",
    "yellow orange": "yellow",
    "orange yellow": "orange",
    "blue green": "blue",
    "red orange": "red",
    "green yellow": "green",
    "violet red": "red",
    "grey": "gray",
    "blue violet": "purple",
    "orange red": "orange",
    "yellow ochre": "yellow"
}

app = Flask(__name__)

PROCESSED_IMAGE_DIR = 'images'
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)

# Load TFLite model
TFLITE_MODEL_PATH = "pattern_recognition_model.tflite"
if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(f"TFLite model file not found: {TFLITE_MODEL_PATH}")

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details for TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['checkered', 'dotted', 'floral', 'solid', 'striped', 'zigzag']

@app.route('/')
def home():
    return 'Welcome!'

def get_closest_color_name(rgb_color):
    if all(value <= 60 for value in rgb_color):
        return "black"

    if all(value >= 240 for value in rgb_color):
        return "white"

    color_info = get_color_from_rgb(rgb_color)
    color_family = color_info['color_family'].lower()
    return base_color_mapping.get(color_family, color_family)

def predict_pattern(pil_image):
    # Convert PIL Image to input size and normalize
    img_array = np.array(pil_image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Set input tensor for TFLite model
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = classes[np.argmax(predictions)]  # Predicted class
    confidence = np.max(predictions) * 100  # Confidence score
    return predicted_class, confidence


@app.route('/process-image', methods=['POST'])
def process_image_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_bytes = BytesIO(image_file.read())
    pil_image = Image.open(image_bytes).convert("RGB")
    img_rgb = np.array(pil_image)

    img_resized = cv2.resize(img_rgb, (480, 640))
    pixels = img_resized.reshape(-1, 3)
    k = 5
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    color_counts = np.bincount(labels.flatten())
    sorted_indices = np.argsort(color_counts)[::-1]
    dominant_colors = [centers[i] for i in sorted_indices[:3]]
    dominant_color_names = [get_closest_color_name(tuple(color.astype(int))) for color in dominant_colors]

    pattern, confidence = predict_pattern(pil_image)
    print(f"Pattern: {pattern}, Confidence: {confidence:.2f}%")

    dominant_colors_list = list(dict.fromkeys(dominant_color_names))
    str_dominant_colors_list = ", ".join(dominant_colors_list)

    return jsonify({
        'dominant_colors': dominant_colors_list,
        'pattern': pattern,
        'pattern_confidence': f"{confidence:.2f}%"
    })


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
