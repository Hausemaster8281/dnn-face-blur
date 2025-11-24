import cv2
import sys
import numpy

def main():
    # A pre-trained Haar Cascade classifier for face detection is loaded (xml provided by OPENCV
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
