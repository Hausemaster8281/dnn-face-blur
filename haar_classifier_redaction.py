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
        # Loop around detected faces, apply blur
        for (x, y, w, h) in faces:
            # region of interest (ROI) extraction, aka, face extraction
            face_roi = frame[y:y+h, x:x+w]

            # Strong gaussian blur appliad to face (within ROI) - slight tweaking of kernel size based on face size
            k_w = w // 3 | 1 # Preferably odd (similar to classifier algorithms)
            k_h = h // 3 | 1
            blurred_face = cv2.GaussianBlur(face_roi, (k_w, k_h), 30)

            # Replaces original ROI, with the blurred version
            frame[y:y+h, x:x+w] = blurred_face

            # Draw a rectangle around the face, for explainability
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

