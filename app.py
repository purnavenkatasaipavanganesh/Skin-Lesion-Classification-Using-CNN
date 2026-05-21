import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- Load Model --------
model = load_model("resnet50_best.h5", compile=False)

# -------- Class Mapping (FULL + SHORT) --------
CLASS_MAP = {
    0: {"short": "akiec", "full": "Actinic Keratosis"},
    1: {"short": "bcc",   "full": "Basal Cell Carcinoma"},
    2: {"short": "bkl",   "full": "Benign Keratosis"},
    3: {"short": "df",    "full": "Dermatofibroma"},
    4: {"short": "mel",   "full": "Melanoma"},
    5: {"short": "nv",    "full": "Melanocytic Nevus"},
    6: {"short": "vasc",  "full": "Vascular Lesion"}
}

LAST_CONV_LAYER = "conv5_block3_out"

# -------- Grad-CAM --------
def make_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# -------- Routes --------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = short_name = confidence = img_path = heatmap_path = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No image uploaded")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="Please select an image")

        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        confidence = float(np.max(preds))

        prediction = CLASS_MAP[class_idx]["full"]
        short_name = CLASS_MAP[class_idx]["short"]

        # ---- Grad-CAM ----
        heatmap = make_gradcam(img_array, model, LAST_CONV_LAYER)
        heatmap = cv2.resize(heatmap, img.size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.imread(img_path)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        heatmap_path = img_path.replace(".", "_gradcam.")
        cv2.imwrite(heatmap_path, overlay)

    return render_template(
        "index.html",
        prediction=prediction,
        short_name=short_name,
        confidence=confidence,
        img_path=img_path,
        heatmap_path=heatmap_path
    )

if __name__ == "__main__":
    app.run(debug=True)
