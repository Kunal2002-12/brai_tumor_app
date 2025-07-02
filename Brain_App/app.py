import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from flask import Flask, render_template, request

app = Flask(__name__)
model = load_model('BrainTumor10EpochsCategorical.h5')
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(filepath):
    try:
        image = cv2.imread(filepath)

        if image is None:
            return "❌ Not a valid image. Please upload a proper picture."

        # Check if image size is too small or completely black/white
        if image.shape[0] < 50 or image.shape[1] < 50:
            return "❌ Image too small. Please upload a clearer brain scan."

        # Convert to grayscale and check average intensity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_pixel = np.mean(gray)

        if avg_pixel < 10 or avg_pixel > 245:
            return "❌ Image is too dark or too bright to be a valid brain scan."

        # Resize and preprocess
        img = Image.fromarray(image)
        img = img.resize((64, 64))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        input_img = input_img / 255.0

        prediction = model.predict(input_img)
        result = np.argmax(prediction, axis=1)[0]

        return "✅ Brain Tumor Detected" if result == 1 else "✅ No Brain Tumor Detected"

    except Exception as e:
        print("Error:", e)
        return "❌ Could not process the image. Make sure it is a valid brain scan."
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = predict_image(filepath)
            return render_template('index.html', result=result, image_url=filepath)
    return render_template('index.html', result=None, image_url=None)

if __name__ == '__main__':
    app.run(debug=True)
