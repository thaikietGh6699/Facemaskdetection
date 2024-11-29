from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import sys

# Reconfigure standard output to handle utf-8 correctly
sys.stdout.reconfigure(encoding='utf-8')

def load_face_detector(prototxtPath, weightsPath):
    """ Load the pre-trained face detector model """
    return cv2.dnn.readNet(prototxtPath, weightsPath)

def load_mask_detector(model_path):
    """ Load the pre-trained mask detection model """
    return load_model(model_path)

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (128, 128), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] # độ tin cậy của mô hình faceNet

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Cắt khuôn mặt ra khỏi khung hình và chuyển đổi từ BGR sang RGB
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (128, 128))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            
            # vị trí khuôn mặt
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        # Dự đoán
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def process_video(video_stream, faceNet, maskNet):
    """ Process video stream, detect faces, and predict mask status """
    while True:
        frame = video_stream.read()
        frame = imutils.resize(frame, width=800)
        # Detect faces and predict mask status
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # Loop over the detected face locations and corresponding predictions
        for (box, pred) in zip(locs, preds):
            # Unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # Determine the class label and color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Display the label and bounding box on the frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Show the output frame
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to quit
            break

def main():
    """ Main function to set up and run the mask detection """
    # Paths to the models
    prototxtPath = r"E:\Face-Mask-Detection\face_detector\deploy.prototxt"
    weightsPath = r"E:\Face-Mask-Detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    maskNetPath = r"E:\Face-Mask-Detection\model_mask_finetuned.h5"

    # Load models
    print("[INFO] loading face detector model...")
    faceNet = load_face_detector(prototxtPath, weightsPath)

    print("[INFO] loading mask detector model...")
    maskNet = load_mask_detector(maskNetPath)

    # Initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    # Process video stream
    process_video(vs, faceNet, maskNet)

    # Cleanup
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
