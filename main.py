import cv2
import sys
from detection import FaceDetector
from processing import FaceAnonymizer
from config import CONFIDENCE_THRESHOLD

def main():
    detector = FaceDetector()
    anonymizer = FaceAnonymizer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        sys.exit(1)

    print(f"Face Anonymizer (DNN), threshold {CONFIDENCE_THRESHOLD}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        faces = detector.detect_faces(frame)
        
        for face in faces:
            frame = anonymizer.blur_face(frame, face['box'])
            anonymizer.draw_ui(frame, face['box'], face['confidence'])

        cv2.imshow('Face Anonymizer (DNN) - Press q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
