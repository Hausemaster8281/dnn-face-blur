import cv2
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROTOTXT_PATH = os.path.join(MODELS_DIR, 'deploy.prototxt')
MODEL_PATH = os.path.join(MODELS_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

# Detection settings
CONFIDENCE_THRESHOLD = 0.6
BLUR_KERNEL_RATIO = 3
BLUR_SIGMA = 30
