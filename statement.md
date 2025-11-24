# Face Anonymizer Project

## Problem Statement
In an era of increasing digital surveillance and video content creation, privacy concerns are paramount. Individuals often appear in videos without consent, or sensitive identities need to be protected in public footage. Manual redaction of faces is time-consuming and prone to error. There is a need for an automated, real-time system that can detect and anonymize faces in video streams to ensure privacy compliance.

## Scope of the Project
This project aims to build a real-time computer vision application that:
1.  Captures live video feed from a webcam.
2.  Automatically detects human faces using Deep Learning (DNN) techniques.
3.  Applies a dynamic blur filter to anonymize the detected faces.
4.  Displays the processed video stream to the user.

The system is designed to be lightweight, running locally on consumer hardware without sending data to the cloud, ensuring maximum privacy.

## Target Users
*   **Content Creators**: Vloggers or streamers who need to anonymize bystanders in public settings.
*   **Journalists**: To protect the identity of interviewees or sources.
*   **Security Personnel**: For monitoring feeds where privacy regulations (like GDPR) require face redaction.
*   **Privacy Advocates**: Individuals interested in personal privacy tools.

## High-Level Features
*   **Real-time Face Detection**: Utilizes a ResNet-10 Single Shot Detector (SSD) model for robust detection across various angles and lighting conditions.
*   **Dynamic Anonymization**: Applies a Gaussian blur that scales with the size of the face to ensure effective redaction.
*   **Confidence Visualization**: Displays the detection confidence score and bounding box for system transparency.
*   **Configurable Thresholds**: Allows users to adjust detection sensitivity to balance between false positives and missed detections.
