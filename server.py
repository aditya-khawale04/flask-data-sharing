import time
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

def background_task():
    count = 0
    while True:
        socketio.emit('new_data', {'value': count})
        time.sleep(1)  # emit every second
        count += 1

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(target=background_task)
    socketio.run(app, host='0.0.0.0', port=3000, debug=True)
