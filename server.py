import cv2
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Replace with your actual IP camera stream URL
# Examples:
# rtsp://username:password@ip_address:port/path
# http://ip_address:port/video
# CAMERA_URL = "http://172.19.216.204/"  # for IP webcam app
CAMERA_URL = 0  # for local webcam

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Failed to connect to camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': frame_b64})
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(target=generate_frames)
    socketio.run(app, host='0.0.0.0', port=3000)
