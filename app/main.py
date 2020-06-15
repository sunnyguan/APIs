import base64
from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

@app.route("/")
def home_view():
    return "<h1>Hello World!</h1>"

@app.route("/api/face", methods=['POST'])
def detect_face():
    try:
        req = request.json

        if req.get('image') is None:
            raise InvalidException('image is required.')

        # decode base64 string into np array
        nparr = np.frombuffer(base64.b64decode(req['image'].encode('utf-8')), np.uint8)

        # decoded image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise InvalidException('Unable to parse the image.')

        num, data = detect(img)
        response = {
            'success': True,
            'status code': 201,
            'message': '{} faces detected'.format(num),
            'data': {'image': data},
            }
        resp = jsonify(response)
        resp.status_code = 200
        return resp

    except Exception as e:
        response = {
            'success': False,
            'status code': 500,
            'message': str(e),
            }
        resp = jsonify(response)
        resp.status_code = 500
        return resp

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.imwrite("test.jpg", image)

    string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
    return len(faces), string