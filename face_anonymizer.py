import cv2
import sys
import numpy

def main():
    # Defined paths to the model files
    prototxt_path = "models/deploy.prototxt"
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

    # Confidence threshold for faces, lower values might cause false-positives
    CONFIDENCE_THRESHOLD = 0.6

    # Loading logic and validation
    print("Loading model...")
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    except cv2.error as e:
        print(f"Error loading DNN: {e}")
        print("Please ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' exist in the 'models' directory, along with proper user-group access.")
        sys.exit(1)

    # Initialize video capture from webcam 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        sys.exit(1)

    print(f"Face Anonymizer (DNN) , threshold {CONFIDENCE_THRESHOLD}. Press 'q' to quit.")

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Acquire frame domensions
        (h, w) = frame.shape[:2]

        # Construct a blob from the image, reconstructing it to 300x300 and perform mean subtraction with (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network for obtaining detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence/probability associated with the prediction
            confidence = detections[0, 0, i, 2]
