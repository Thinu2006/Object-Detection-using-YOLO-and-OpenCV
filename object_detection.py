import cv2
import numpy as np

# Initialize video capture (0 for default webcam)
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load the YOLO model
weights_path = 'D:/ColourDetection/yolov4.weights'
config_path = 'D:/ColourDetection/yolov4.cfg'
net = cv2.dnn.readNet(weights_path, config_path)

# Load class labels
with open('D:/ColourDetection/yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Prepare the frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()

    # Run the YOLO model on the frame
    layer_outputs = net.forward(output_layers)

    bbox, labels, confidences = [], [], []

    # Process YOLO output
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = map(int, detection[0:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                bbox.append([x, y, w, h])
                labels.append(classes[class_id])
                confidences.append(float(confidence))

    # Non-maxima suppression
    indices = cv2.dnn.NMSBoxes(bbox, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = bbox[i]
            label = labels[i]
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Show the frame
    cv2.imshow("Object Detection", frame)
    key = cv2.waitKey(1)

    if key == 27:  # Press ESC to exit
        break

video.release()
cv2.destroyAllWindows()


