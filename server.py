from flask import Flask, render_template, request
from flask_socketio import SocketIO
import cv2
import base64
import time
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import threading
import os
from dotenv import load_dotenv

# Optional alarm sound
try:
    from playsound import playsound
except:
    playsound = None

# Load Twilio credentials from .env
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = '+17697596183'
TWILIO_TO = '+919156662372'

try:
    from twilio.rest import Client
    twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
except:
    twilio_client = None

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Load YOLOv5 model
model = YOLO("yolov5su.pt")
PERSON_CLASS_ID = 0
CAMERA_URL = 0  # Use 0 for default webcam

# Default polygon
default_zone_coords = [(100, 100), (500, 100), (500, 400), (100, 400)]
default_zone_polygon = Polygon(default_zone_coords)

# Dynamic polygon state
user_polygon_coords = None
user_polygon_polygon = None
use_user_polygon = False

# Alert states
alert_sent = False
alarm_triggered = False

def buzz_alarm():
    if playsound and os.path.exists("alarm.mp3"):
        playsound("alarm.mp3")

def make_phone_call():
    if twilio_client:
        try:
            call = twilio_client.calls.create(
                to=TWILIO_TO,
                from_=TWILIO_FROM,
                twiml='<Response><Say>This is an emergency alert. A person has entered the danger zone.</Say></Response>'
            )
            print("Call initiated:", call.sid)
        except Exception as e:
            print("Call failed:", e)

@socketio.on('sos_triggered')
def handle_sos_triggered(data):
    """Handle SOS alert triggered after 5-second countdown"""
    message = data.get('message', 'SOS Alert')
    location = data.get('location', None)
    
    print(f"🚨 {message}")
    if location:
        print(f"Location: {location['latitude']}, {location['longitude']}")
    
    # Trigger emergency actions
    threading.Thread(target=buzz_alarm).start()
    threading.Thread(target=make_phone_call).start()
    
    # Broadcast SOS alert to all connected clients
    socketio.emit('sos_alert', {
        'message': message,
        'location': location,
        'timestamp': time.time()
    })

@socketio.on('update_polygon')
def handle_polygon_update(data):
    global user_polygon_coords, user_polygon_polygon, use_user_polygon
    points = data.get('points', [])
    if len(points) >= 3:
        coords = [(int(p['x']), int(p['y'])) for p in points]
        user_polygon_coords = coords
        user_polygon_polygon = Polygon(coords)
        use_user_polygon = True
        print("Updated polygon:", coords)

@socketio.on('clear_polygon')
def handle_polygon_clear(_):
    global user_polygon_coords, user_polygon_polygon, use_user_polygon
    user_polygon_coords = None
    user_polygon_polygon = None
    use_user_polygon = False
    print("Polygon cleared")

def detect_and_stream():
    global alert_sent, alarm_triggered
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

        # Determine polygon
        current_polygon = user_polygon_polygon if use_user_polygon else default_zone_polygon
        current_coords = user_polygon_coords if use_user_polygon else default_zone_coords

        if current_coords:
            cv2.polylines(frame, [np.array(current_coords, np.int32)], True, (255, 255, 0), 2)

        if detections is not None:
            for box in detections:
                cls = int(box.cls[0])
                if cls == PERSON_CLASS_ID:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    point = Point(center_x, center_y)

                    inside = current_polygon.contains(point)
                    color = (0, 255, 0) if inside else (0, 0, 255)
                    label = f'Person {conf:.2f}'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, color, -1)

                    if not inside:
                        alarm_triggered = True

        cv2.putText(frame, f'Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if alarm_triggered:
            cv2.putText(frame, '🚨 ALERT: Person Outside Safe Zone', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not alert_sent:
                threading.Thread(target=buzz_alarm).start()
                threading.Thread(target=make_phone_call).start()
                alert_sent = True
        else:
            alert_sent = False

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

@app.route('/call', methods=['POST'])
def call_sos():
    threading.Thread(target=make_phone_call).start()
    return {'status': 'SOS call triggered'}

if __name__ == '__main__':
    socketio.start_background_task(detect_and_stream)
    socketio.run(app, host='0.0.0.0', port=3000)
