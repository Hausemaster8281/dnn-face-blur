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

