# Face Anonymizer (DNN)

A real-time privacy tool that automatically detects and blurs faces in live video streams using Deep Learning.

## Overview
This project implements a robust face anonymization system using OpenCV and Python. Unlike traditional Haar Cascade methods, this solution utilizes a Deep Neural Network (ResNet-10 SSD) to ensure high accuracy even with tilted faces, rapid motion, or varying lighting conditions.

## Features
*   **Deep Learning Detection**: Uses a Caffe-based ResNet-10 model for superior accuracy.
*   **Real-Time Processing**: Optimized for live webcam feeds.
*   **Dynamic Blurring**: Blur intensity adjusts automatically based on face size.
*   **Modular Design**: Clean architecture separating detection, processing, and configuration.

## Technologies Used
*   **Python 3.x**
*   **OpenCV (cv2)**: For video capture, image processing, and DNN inference.
*   **NumPy**: For matrix operations.

## Installation & Run

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd dnn-face-blur
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python main.py
    ```

## Testing
*   Run the application and sit in front of the webcam.
*   Move your head side-to-side and tilt it to test the DNN robustness.
*   Bring your face closer and further away to verify the dynamic blur scaling.
*   Press **'q'** to exit the application.

## Project Structure
*   `main.py`: Entry point of the application.
*   `detection.py`: Handles loading the DNN model and inference.
*   `processing.py`: Applies blur effects and draws UI elements.
*   `config.py`: Stores configuration constants (paths, thresholds).
*   `models/`: Contains the `deploy.prototxt` and `.caffemodel` files.
