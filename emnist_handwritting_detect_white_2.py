import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
from tkinter import simpledialog, messagebox

# Load label mapping from file
def load_label_mapping(path='emnist-balanced-mapping.txt'):
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            key, val = line.strip().split()
            mapping[int(key)] = chr(int(val))
    return mapping

label_map = load_label_mapping()
reverse_label_map = {v: k for k, v in label_map.items()}

# App class
class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EMNIST Handwriting Recognition GUI")

        # Load model inside the class
        self.model = tf.keras.models.load_model('upmy_model.h5')

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(self.root, text='Predict', command=self.predict_digit)
        self.button_predict.pack()

        self.button_add = tk.Button(self.root, text='Add to Test Set', command=self.add_to_test_set)
        self.button_add.pack()

        self.button_eval = tk.Button(self.root, text='Evaluate Accuracy', command=self.evaluate_accuracy)
        self.button_eval.pack()

        self.button_train = tk.Button(self.root, text='Train on My Samples', command=self.fine_tune_on_user_data)
        self.button_train.pack()

        self.button_clear = tk.Button(self.root, text='Clear', command=self.clear_canvas)
        self.button_clear.pack()

        self.label_result = tk.Label(self.root, text='', font=('Helvetica', 20))
        self.label_result.pack()

        self.image1 = Image.new("L", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind('<B1-Motion>', self.paint)

        self.test_images = []
        self.true_labels = []

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
        image = image.resize((28, 28))
        image = ImageOps.invert(image)
        image_np = np.array(image).astype('float32') / 255.0
        image_np = image_np.reshape(1, 28, 28, 1)
        image_np = np.transpose(image_np, (0, 2, 1, 3))
        return image_np

    def predict_digit(self):
        img = self.image1.copy()
        processed_img = self.preprocess(img)
        prediction = self.model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        predicted_char = label_map.get(predicted_class, '?')
        self.label_result.config(text=f'Prediction: {predicted_char}')

    def add_to_test_set(self):
        actual_char = simpledialog.askstring("Input", "Enter the character you drew (case-sensitive):")
        if actual_char is None or len(actual_char) != 1:
            messagebox.showwarning("Invalid Input", "Please enter a single character.")
            return
        if actual_char not in reverse_label_map:
            messagebox.showerror("Not Found", f"'{actual_char}' not found in label map.")
            return
        img = self.image1.copy()
        processed_img = self.preprocess(img)
        self.test_images.append(processed_img)
        self.true_labels.append(reverse_label_map[actual_char])
        messagebox.showinfo("Added", f"Added '{actual_char}' to test set.")
        self.clear_canvas()

    def evaluate_accuracy(self):
        if not self.test_images:
            messagebox.showwarning("No Data", "No test samples added.")
            return
        X = np.vstack(self.test_images)
        y_true = np.array(self.true_labels)
        preds = self.model.predict(X)
        y_pred = np.argmax(preds, axis=1)
        accuracy = np.mean(y_true == y_pred)
        messagebox.showinfo("Accuracy", f"Custom Drawn Test Set Accuracy: {accuracy * 100:.2f}%")

    def fine_tune_on_user_data(self):
        if not self.test_images:
            messagebox.showwarning("No Data", "No training samples added.")
            return

        X_train = np.vstack(self.test_images)
        y_train = np.array(self.true_labels)

        num_classes = len(label_map)
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)

        self.model.fit(X_train, y_train_cat, epochs=5, batch_size=4, verbose=1)

        self.model.save("user_tuned_model.h5")
        np.savez_compressed("my_drawings_dataset.npz", X=X_train, y=y_train)

        messagebox.showinfo("Training Done", "Model fine-tuned on your inputs and saved.")

# Run the app
if __name__ == '__main__':
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
