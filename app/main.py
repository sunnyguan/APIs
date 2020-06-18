import base64
from flask import Flask, request, jsonify
import numpy as np
import cv2
import json
import os
import atexit
import sys
import requests
import time
import difflib
import argparse
import imutils
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

"""
app = Flask(__name__)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
driver = webdriver.Chrome(chrome_options=chrome_options)
"""

app = Flask(__name__)
CHROMEDRIVER_PATH = "/app/.chromedriver/bin/chromedriver"
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
chrome_bin = os.environ.get('GOOGLE_CHROME_BIN', "chromedriver")
options = webdriver.ChromeOptions()
options.binary_location = chrome_bin
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument('--headless')
# options.add_argument('window-size=1200x600')
driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=options)

def close_running_threads():
    driver.close()
atexit.register(close_running_threads)

@app.route("/hello")
def home_view():
    return "<h1>Hello World!</h1>"

@app.route('/ocr')
def homepage():
    return """
<!DOCTYPE html>
<head>
   <title>My title</title>
   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/kognise/water.css@latest/dist/dark.min.css">
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
                    var p = JSON.parse(this.responseText);
                    var tbl = "<table><tr><th>ID</th><th>Word</th><th>Confidence</th></tr>";
                    for (var key in p) {
                        if (p.hasOwnProperty(key)) {
                            tbl += "<tr><td>" + key + "</td><td>" + p[key]["word"] + "</td><td>" + p[key]["confidence"] + "</td></tr>";
                        }
                    }
                    tbl += "</table>";
                    document.body.innerHTML += tbl;
                    console.log(tbl)
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
        <img id="sample" src="http://stash.compjour.org/assets/images/sunset.jpg" alt="it's a nice sunset">
    </a>
    <input type="file" id="img" name="img" accept="image/*" onchange="document.getElementById('sample').src = window.URL.createObjectURL(this.files[0])">
    <button onclick="getText()">Submit</button>
</body>
"""

@app.route('/')
def testpage():
    return """
<!DOCTYPE html>
<head>
   <title>My title</title>
   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/kognise/water.css@latest/dist/dark.min.css">
   <script>
        function getText() {
            document.body.innerHTML += "request submitted, please wait...";
            var query = document.getElementById("srch").value;
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/course', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = function () {
                var p = JSON.parse(this.responseText);
                document.body.innerHTML += "check console for response";
                console.log(p)
            };
            var inp = {"query": query};
            xhr.send(JSON.stringify(inp));
        }
   </script>
</head>
<body style="width: 880px; margin: auto;">  
    <input id="srch" type="text" placeholder="Search Term" />
    <button onclick="getText()">Submit</button>
</body>
"""


@app.route("/api/course", methods=['POST'])
def course_api():
    req = request.json
    query = req["query"]

    driver.get("https://coursebook.utdallas.edu/search")
    time.sleep(1)

    driver.find_element_by_id("srch").send_keys(query)
    driver.find_element_by_id("srch").send_keys(Keys.RETURN)
    # time.sleep(3)
    table = 0
    ki = 0
    while ki <= 70:
        try:
            table = driver.find_element_by_xpath("//table/tbody")
            break
        except Exception as e:
            print('wait...')
            ki += 1
            time.sleep(0.3)
    if table == 0:
        resp = jsonify({"bad": "true"})
        resp.status_code = 200
        return resp
    else:
        rows = table.find_elements_by_tag_name("tr")
        text = []
        j = 1
        for row in rows:
            curr = {}
            col = row.find_elements_by_tag_name("td")
            for i in range(0, len(col)):
                curr[i] = col[i].text
            text.append(curr)
            j+=1
        print(text)
        resp = jsonify(text)
        resp.status_code = 200
        return resp

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
        
        # img = transform(img)
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
    
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def transform(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    orig = imutils.resize(image, height = 500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    if len(cnts) > 0:
        screenCnt = cnts[0]
        try:
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            return warped
        except e:
            return image
    else:
        return image