from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import zipfile
import shutil
import os
import sys

# Reconfigure standard output to handle utf-8 correctly
sys.stdout.reconfigure(encoding='utf-8')
app = Flask(__name__)

prototxtPath = r"E:\Face-Mask-Detection\face_detector\deploy.prototxt"
weightsPath = r"E:\Face-Mask-Detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
maskNetPath = r"E:\Face-Mask-Detection\model_mask_finetuned.h5"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(maskNetPath)

def detect_and_predict_mask(image):
    """Phát hiện khuôn mặt và dự đoán khẩu trang"""
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (128, 128), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Cắt khuôn mặt và xử lý
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (128, 128))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Đọc ảnh và dự đoán
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Dự đoán khẩu trang
            (locs, preds) = detect_and_predict_mask(image)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            result_image_path = os.path.join("static", "result_" + file.filename)
            cv2.imwrite(result_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            return render_template("detect.html", filename="result_" + file.filename)

    return render_template("detect.html", filename=None)

@app.route("/fine_tune", methods=["GET", "POST"])
def fine_tune():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Lưu tệp zip tải lên
            zip_path = os.path.join("static", file.filename)
            file.save(zip_path)

            # Giải nén tệp zip
            extract_path = os.path.join("static", "dataset")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            os.remove(zip_path)

            with_mask_path = None
            without_mask_path = None

            for root, dirs, files in os.walk(extract_path):
                if "with_mask" in dirs and "without_mask" in dirs:
                    with_mask_path = os.path.join(root, "with_mask")
                    without_mask_path = os.path.join(root, "without_mask")
                    break

            if with_mask_path is None or without_mask_path is None:
                shutil.rmtree(extract_path)
                return "Dataset does not contain the required 'with_mask' and 'without_mask' folders.", 400

            data, labels = [], []
            categories = ["with_mask", "without_mask"]
            for category in categories:
                category_path = locals()[category + "_path"]
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    image = cv2.imread(img_path)
                    if image is not None:
                        image = cv2.resize(image, (128, 128))
                        image = img_to_array(image)
                        image = preprocess_input(image)
                        data.append(image)
                        labels.append(category)
            
            lb = LabelBinarizer()
            labels = lb.fit_transform(labels)
            data = np.array(data, dtype="float32")
            labels = np.array(labels)

            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

            INIT_LR = 1e-5
            EPOCHS = 10
            BS = 8

            maskNet = load_model(maskNetPath)
            for layer in maskNet.layers:
                layer.trainable = True
                if isinstance(layer, Dense):
                    layer.kernel_regularizer = regularizers.l2(0.01)

            aug = ImageDataGenerator(
                rotation_range=30,
                zoom_range=0.2,
                width_shift_range=0.25,
                height_shift_range=0.25,
                shear_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest",
                preprocessing_function=preprocess_input
            )

            maskNet.compile(
                optimizer=Adam(learning_rate=INIT_LR),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

            history = maskNet.fit(
                aug.flow(X_train, y_train, batch_size=BS),
                steps_per_epoch=len(X_train) // BS,
                validation_data=(X_test, y_test),
                epochs=EPOCHS,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

            preds = maskNet.predict(X_test, batch_size=BS)
            y_pred = np.argmax(preds, axis=1)
            y_true = y_test
            accuracy = accuracy_score(y_true, y_pred)
            classification = classification_report(y_true, y_pred)

            updated_model_path = os.path.join(maskNetPath)
            maskNet.save(updated_model_path)

            shutil.rmtree(extract_path)

            return render_template("fine_tune.html", accuracy=accuracy, classification=classification)

    return render_template("fine_tune.html", accuracy=None, classification=None)

@app.route("/")
def home():
    return render_template("index.html", filename=None)

if __name__ == "__main__":
    app.run(debug=True)
