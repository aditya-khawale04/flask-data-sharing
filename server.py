from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import time
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Load YOLOv5s model (you can also use yolov8n.pt, etc.)
model = YOLO('yolov5s.pt')

# Only detect "person" class (class 0)
PERSON_CLASS_ID = 0

# Replace with your IP camera or video file or webcam
# CAMERA_URL = 'http://192.168.1.100:8080/video'  # for IP Webcam
CAMERA_URL = 0  # for Webcam

def detect_and_stream():
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run detection
        results = model(frame, verbose=False)
        detections = results[0].boxes  # Get detected boxes

        person_count = 0

        if detections is not None:
            for box in detections:
                cls = int(box.cls[0])
                if cls == PERSON_CLASS_ID:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f'Person {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show person count on top
        cv2.putText(frame, f'Persons: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Encode and send frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': frame_b64, 'count': person_count})

        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(target=detect_and_stream)
    socketio.run(app, host='0.0.0.0', port=3000)
