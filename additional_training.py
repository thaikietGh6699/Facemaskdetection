from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

sys.stdout.reconfigure(encoding='utf-8')

model_path = "model_mask_finetuned.h5"
DIRECTORY = r"E:\Face_Mask_Detection_Dataset_MaskNet"
CATEGORIES = ["with_mask", "without_mask"]

# Hyperparameters
INIT_LR = 1e-5
EPOCHS = 10
BS = 8

# Load pre-trained model
print("[INFO] loading model...")
model = load_model(model_path)

# Unfreeze all layers and add regularization
for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, Dense):
        layer.kernel_regularizer = regularizers.l2(0.01) 

# Data augmentation
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

# Load and preprocess images
print("[INFO] loading images...")
data, labels = [], []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(128, 128))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

# One-hot encoding labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Convert to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Compile model with adjusted learning rate
model.compile(optimizer=Adam(learning_rate=INIT_LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callbacks: EarlyStopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Train the model
print("[INFO] fine-tuning model on new data...")
history = model.fit(
    aug.flow(X_train, y_train, batch_size=BS), 
    validation_data=(X_test, y_test), 
    epochs=EPOCHS, 
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
print("[INFO] evaluating model on new data...")
preds = model.predict(X_test, batch_size=BS)

y_true = y_test
y_pred = np.argmax(preds, axis=1)

print("[INFO] classification report:")
print(classification_report(y_true, y_pred))

print("[INFO] confusion matrix:")
print(confusion_matrix(y_true, y_pred))

# Save the model
model.save(model_path)
print("[INFO] model updated and saved!")

