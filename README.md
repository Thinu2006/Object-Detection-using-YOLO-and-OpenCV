# Object Detection using YOLO and OpenCV

## Overview
This project performs real-time object detection using the YOLO (You Only Look Once) deep learning model with OpenCV and a webcam. The script captures video from the webcam, processes frames through the YOLO model, and identifies objects by drawing bounding boxes around detected objects.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)

Install the required dependencies using:
```bash
pip install opencv-python numpy
```

## Model Files
This script requires the YOLO model files:
- `yolov4.weights`: Pre-trained weights for YOLOv4
- `yolov4.cfg`: Configuration file for YOLOv4
- `yolov3.txt`: File containing class labels for YOLO

Ensure these files are placed in `D:/ColourDetection/` or update the script with the correct file paths.

## How to Run
1. Ensure your webcam is connected and accessible.
2. Place the YOLO model files in the correct directory.
3. Run the script using:
   ```bash
   python object_detection.py
   ```
4. Press `ESC` to exit the program.

## How It Works
1. The script initializes webcam capture.
2. The YOLO model is loaded with pre-trained weights and configuration.
3. Frames from the webcam are processed through YOLO for object detection.
4. Bounding boxes and labels are drawn for detected objects with confidence scores above 50%.
5. The processed frame is displayed in a window.
6. Press `ESC` to stop execution and close the webcam.

## Notes
- Adjust the `confidence` threshold (default `0.5`) if needed for more/less strict detection.
- Make sure the `yolov3.txt` file contains the correct class labels for the model being used.
- If the webcam is inaccessible, check your device's camera permissions or update the `cv2.VideoCapture(0)` parameter if multiple cameras are available.

## License
This project is open-source. Feel free to modify and use it for learning purposes!

