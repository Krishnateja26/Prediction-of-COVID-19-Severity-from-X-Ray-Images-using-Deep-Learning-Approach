from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io

img_size = 100

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the model
model = load_model("my_model_Covid.keras")
print("Model loaded")

label_dict = {0: 'Covid19 Negative', 1: 'Covid19 Positive'}

def preprocess(img):
    img = np.array(img.convert('L'))  # Convert to grayscale
    img = img / 255.0
    resized = cv2.resize(img, (img_size, img_size))
    reshaped = resized.reshape(1, img_size, img_size, 1)  # Add channel dimension
    return reshaped

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about-us.html")
def about():
    return render_template("About_us.html")

@app.route("/corona-virus.html")
def corona():
    return render_template("Corona Virus.html")

@app.route("/deep-learning.html")
def deeplearning():
    return render_template("Deep Learning.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.get_json(force=True)
        if 'image' not in message:
            return jsonify({'error': 'Image data is missing'}), 400

        encoded = message['image']
        decoded = base64.b64decode(encoded)
        dataBytesIO = io.BytesIO(decoded)
        dataBytesIO.seek(0)
        image = Image.open(dataBytesIO)

        test_image = preprocess(image)
        prediction = model.predict(test_image)
        result = np.argmax(prediction, axis=1)[0]
        accuracy = float(np.max(prediction, axis=1)[0])
        label = label_dict[result]

        response = {'prediction': {'result': label, 'accuracy': accuracy}}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)






