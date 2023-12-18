import socketio
import eventlet
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server()

app = Flask(__name__) 


speed_limit = 15

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img/255
    return img

@sio.on('connect') # decorator
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])
        steering_angle = float(model.predict(image))
        speed = float(data['speed'])
        
        throttle = min((1.0 - abs(steering_angle), (1-speed/speed_limit)))
        send_control(steering_angle, throttle)

    
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })



if __name__ == '__main__':
    model = load_model('alpha_model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)