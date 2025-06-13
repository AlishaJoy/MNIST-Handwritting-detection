import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2
import tensorflow as tf

# Load your trained model (ensure same architecture)
model = tf.keras.models.load_model('upmy_model.h5')  # Save your trained model earlier

# Class mapping for EMNIST Balanced
# Use the emnist-balanced-mapping.txt file to map indices to characters
# Download from: https://www.nist.gov/system/files/documents/2017/11/03/emnist-dataset-mapping.txt
def load_label_mapping(path='emnist-balanced-mapping.txt'):
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            key, val = line.strip().split()
            mapping[int(key)] = chr(int(val))
    return mapping

label_map = load_label_mapping()

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Character (EMNIST)")

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(self.root, text='Predict', command=self.predict_digit)
        self.button_predict.pack()

        self.label_result = tk.Label(self.root, text='', font=('Helvetica', 20))
        self.label_result.pack()

        self.button_clear = tk.Button(self.root, text='Clear', command=self.clear_canvas)
        self.button_clear.pack()

        self.image1 = Image.new("L", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)

        self.canvas.bind('<B1-Motion>', self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, 280, 280], fill='white')
        self.label_result.config(text='')

    def preprocess(self, image):
        # Resize to 28x28 and invert (EMNIST uses white-on-black)
        image = image.resize((28, 28))
        image = ImageOps.invert(image)

        image_np = np.array(image).astype('float32') / 255.0
        image_np = image_np.reshape(1, 28, 28, 1)
        image_np = np.transpose(image_np, (0, 2, 1, 3))  # Match EMNIST rotated format
        return image_np

    def predict_digit(self):
        img = self.image1.copy()
        processed_img = self.preprocess(img)
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        predicted_char = label_map.get(predicted_class, '?')

        self.label_result.config(text=f'Prediction: {predicted_char}')

if __name__ == '__main__':
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
