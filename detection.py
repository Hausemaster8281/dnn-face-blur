import cv2
import numpy as np
import sys
from config import PROTOTXT_PATH, MODEL_PATH, CONFIDENCE_THRESHOLD

class FaceDetector:
    def __init__(self):
        print("Loading model...")
        try:
            self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        except cv2.error as e:
            print(f"Error loading DNN: {e}")
            print("Please ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' exist in the 'models' directory.")
            sys.exit(1)

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Clip to frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                
                faces.append({
                    'box': (startX, startY, endX, endY),
                    'confidence': confidence
                })
        return faces
