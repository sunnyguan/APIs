import base64
from flask import Flask, request, jsonify
import numpy as np
import cv2
import json
import os
import sys
import requests
import time
import difflib

app = Flask(__name__)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

@app.route("/hello")
def home_view():
    return "<h1>Hello World!</h1>"

@app.route('/')
def homepage():
    return """
<!DOCTYPE html>
<head>
   <title>My title</title>
   <link rel="stylesheet" href="http://stash.compjour.org/assets/css/foundation.css">
   <script>
        function getText() {
            var file = document.getElementById("img").files[0];
            var reader = new FileReader();
            reader.onloadend = function() {
                console.log('RESULT', reader.result)
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/api/face', true);
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.onload = function () {
                    // do something to response
                    console.log(this.responseText);
                };
                var inp = {"image": reader.result.split(',')[1]};
                xhr.send(JSON.stringify(inp));
            }
            reader.readAsDataURL(file);
        }
   </script>
</head>
<body style="width: 880px; margin: auto;">  
    <h1>Visible stuff goes here</h1>
    <p>here's a paragraph, fwiw</p>
    <p>And here's an image:</p>
    <a href="https://www.flickr.com/photos/zokuga/14615349406/">
        <img src="http://stash.compjour.org/assets/images/sunset.jpg" alt="it's a nice sunset">
    </a>
    <input type="file" id="img" name="img" accept="image/*">
    <button onclick="getText()">Submit</button>
</body>
"""

@app.route("/api/face", methods=['POST'])
def detect_face():
    try:
        req = request.json

        # decode base64 string into np array
        nparr = np.frombuffer(base64.b64decode(req['image'].encode('utf-8')), np.uint8)

        # decoded image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return 'Unable to parse the image.'

        """num, data = detect(img)
        response = {
            'success': True,
            'status code': 201,
            'message': '{} faces detected'.format(num),
            'data': {'image': data},
            }
        """

        strs = imgToConcat(img)
        text = getWords()

        resp = jsonify(text)
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
    # cv2.imwrite("test.jpg", image)

    string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
    return len(faces), string

def imgToConcat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1408, 1697)) 

    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    img_bin = 255-img_bin 

    kernel_length = np.array(img).shape[1]//85
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    alpha = 0.5
    beta = 1.0 - alpha
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contours: cv2.boundingRect(contours)[0] + int(cv2.boundingRect(contours)[1] / 50.0) * 50 * img.shape[1] )

    idx = 0
    cropped_imgs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h > 37 and h < 45 and w > 100):
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cropped_imgs.append(new_img)

    y = 0
    height = sum(image.shape[0] for image in cropped_imgs)
    width = max(image.shape[1] for image in cropped_imgs)
    output = np.zeros((height,width))

    for image in cropped_imgs:
        h,w = image.shape
        output[y:y+h,0:w] = image
        y += h
    cv2.imwrite("concatted2.jpg", output)
    string = base64.b64encode(cv2.imencode('.jpg', output)[1]).decode()
    return string

def getWords():
    endpoint = "https://homeworkocr.cognitiveservices.azure.com/" # os.environ['COMPUTER_VISION_ENDPOINT'] # https://homeworkocr.cognitiveservices.azure.com/
    subscription_key = "a87c544e2d874de5a8b3eb6c92122482" # os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY'] # a87c544e2d874de5a8b3eb6c92122482

    text_recognition_url = endpoint + "/vision/v3.0/read/analyze"
    image_data = open("concatted2.jpg", "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-Type': 'application/octet-stream'}
    response = requests.post(text_recognition_url, headers=headers, data=image_data)
    response.raise_for_status()

    operation_url = response.headers["Operation-Location"]
    print("response 1: " + operation_url)
    analysis = {}
    poll = True
    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        
        print(json.dumps(analysis, indent=4))

        time.sleep(0.5)
        if ("analyzeResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'failed'):
            poll = False

    key = ['roughly symetrical', 'aproximately', 'is greater than', 'is less than', 'scaled', 'labeled', 'typical', 'mean', 'distribution', 'appoximately Normal', 'bias', 'sampling', 'random assignment', 'group']

    text = {}
    idx = 0
    if ("analyzeResult" in analysis):
        for line in analysis["analyzeResult"]["readResults"][0]["lines"]:
            seq = difflib.SequenceMatcher(None,key[idx],line["text"])
            conf = 0
            for word in line["words"]:
                conf += word["confidence"]
            conf = conf / len(line["words"])
            # print(line["text"] + " --- confidence: " + str(conf) + ", distance: " + str(seq.quick_ratio() * 100))
            text[idx] = {"word": line["text"], "confidence":str(conf)}
            idx += 1
    return text