import cv2
import sys
import numpy

def main():
    # A pre-trained Haar Cascade classifier for face detection is loaded (xml provided by OPENCV
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Face cascade classifier failure.")
        sys.exit(1)
    # Initialize video capture from the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        sys.exit(1)

    print("Face Anonymizer (Haar Cascade) started. Press 'q' to quit.")
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break
        # Convert the frame to grayscale, for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect faces in frame. ScaleFactor(1) = no size reduction on image, minNeighbors(5) minimum numbers contined by rectangle of candidate to retain it
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

