from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model and mapping
MODEL_PATH = 'my_model2.h5'
MAPPING_PATH = 'emnist-balanced-mapping.txt'
model = tf.keras.models.load_model(MODEL_PATH)
mapping = pd.read_csv(
    MAPPING_PATH,
    delim_whitespace=True,
    header=None,
    index_col=0
).squeeze()



def annotate_and_predict(image_path):
    # Read and preprocess
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours top-to-bottom, left-to-right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: (b[0][1], b[0][0]))]

    result_str = ''
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Crop and resize
        digit = thresh[y:y+h, x:x+w]
        resized = cv2.resize(digit, (20, 20))
        padded = np.pad(resized, ((4,4),(4,4)), 'constant', constant_values=0)
        normalized = padded.astype('float32') / 255.0

        # Predict
        preds = model.predict(normalized.reshape(1, 28, 28, 1))
        idx = np.argmax(preds)
        ascii_code = mapping.iloc[idx]
        char = chr(ascii_code)
        result_str += char

        # Annotate image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            image,
            char,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    # Encode annotated image to display in HTML
    _, buffer = cv2.imencode('.png', image)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return result_str, encoded


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_data = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result, img_data = annotate_and_predict(filepath)

    return render_template('index.html', result=result, img_data=img_data)


if __name__ == '__main__':
    app.run(debug=True)