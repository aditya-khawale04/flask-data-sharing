from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import time
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import threading
import os

try:
    from playsound import playsound
except:
    playsound = None  # Handle later

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Load YOLOv5s model
model = YOLO("yolov5su.pt")
PERSON_CLASS_ID = 0

# Your IP camera or local webcam or video file
# CAMERA_URL = 'http://192.168.1.100:8080/video'  # IP Webcam app
CAMERA_URL = 0  # For laptop webcam

# Define polygon coordinates (clockwise) - adjust based on your frame
safe_zone_coords = [(100, 100), (500, 100), (500, 400), (100, 400)]
safe_zone_polygon = Polygon(safe_zone_coords)

alarm_triggered = False

def buzz_alarm():
    if playsound and os.path.exists("alarm.mp3"):
        playsound("alarm.mp3")  # Add your alarm.mp3 file in same directory

def detect_and_stream():
    global alarm_triggered
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)
        detections = results[0].boxes
        person_count = 0
        alarm_triggered = False

        # Draw safe zone
        cv2.polylines(frame, [np.array(safe_zone_coords, np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)

        if detections is not None:
            for box in detections:
                cls = int(box.cls[0])
                if cls == PERSON_CLASS_ID:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    point = Point(center_x, center_y)
                    inside = safe_zone_polygon.contains(point)

                    if not inside:
                        alarm_triggered = True
                        color = (0, 0, 255)  # Red for danger
                    else:
                        color = (0, 255, 0)  # Green for safe

                    label = f'Person {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, color, -1)

        # Person count and alarm info
        cv2.putText(frame, f'Persons: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if alarm_triggered:
            cv2.putText(frame, '🚨 ALERT: Person Outside Safe Zone', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            threading.Thread(target=buzz_alarm).start()

        # Encode and emit
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('video_frame', {
            'image': frame_b64,
            'count': person_count,
            'alert': alarm_triggered
        })

        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(target=detect_and_stream)
    socketio.run(app, host='0.0.0.0', port=3000)
