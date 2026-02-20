from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("knn_mnist_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]

        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_GRAYSCALE
        )

        # Resize large image first
        img = cv2.resize(img, (200, 200))

        # Auto invert if needed
        if img.mean() > 127:
            img = cv2.bitwise_not(img)

        # Threshold
        _, img = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find contours
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            x, y, w, h = cv2.boundingRect(
                max(contours, key=cv2.contourArea)
            )
            digit = img[y:y+h, x:x+w]
        else:
            digit = img

        # Resize digit to 20x20
        digit = cv2.resize(digit, (20, 20))

        # Put digit in center of 28x28
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = digit

        # Normalize
        canvas = canvas / 255.0
        canvas = canvas.reshape(1, -1)

        prediction = model.predict(canvas)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)