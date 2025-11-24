import cv2
from config import BLUR_KERNEL_RATIO, BLUR_SIGMA

class FaceAnonymizer:
    def blur_face(self, frame, face_box):
        (startX, startY, endX, endY) = face_box
        face_roi = frame[startY:endY, startX:endX]
        
        if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
            k_w = (endX - startX) // BLUR_KERNEL_RATIO | 1
            k_h = (endY - startY) // BLUR_KERNEL_RATIO | 1
            blurred_face = cv2.GaussianBlur(face_roi, (k_w, k_h), BLUR_SIGMA)
            frame[startY:endY, startX:endX] = blurred_face
        return frame

    def draw_ui(self, frame, face_box, confidence):
        (startX, startY, endX, endY) = face_box
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
