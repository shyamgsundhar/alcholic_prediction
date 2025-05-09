from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained model (e.g., your .h5 model)
model = tf.keras.models.load_model('your_model.h5')  # Replace with your model's path

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    img = request.files['image']
    img = Image.open(BytesIO(img.read()))  # Open image
    img = img.resize((224, 224))  # Resize image to match model input
    img_array = np.array(img) / 255.0  # Normalize image if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    label = "Normal" if prediction[0] > 0.5 else "Alcohol"
    confidence = prediction[0]

    return jsonify({"label": label, "confidence": float(confidence)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run on all IPs of your local network
