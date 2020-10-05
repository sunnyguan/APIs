import base64
from flask import Flask, request, jsonify, flash, redirect, url_for, send_from_directory, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import json
import os
import os.path
import re
import atexit
import sys
import requests
import time
import difflib
import argparse
import imutils
import http.client
import mimetypes
import logging
from itertools import islice
from bs4 import BeautifulSoup as bs4
from PyPDF2 import PdfFileReader
from PIL import Image
from flask import send_from_directory, Response
import PIL
import functools
import urllib.request

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

on_server = False 
chrome_options = None
CHROMEDRIVER_PATH = None
if not on_server:
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
else:
    CHROMEDRIVER_PATH = "/app/.chromedriver/bin/chromedriver"
    chrome_bin = os.environ.get('GOOGLE_CHROME_BIN', "chromedriver")
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = chrome_bin
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36 Edg/85.0.564.51")

def seleniumRefresh():
    global headers
    print("old cookie: " + headers["Cookie"])
    if not on_server:
        driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=r"C:\Eworkspace\FaceAPI\testApp\app\chromedriver\chromedriver.exe")
    else:
        driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)   
    driver.get("https://coursebook.utdallas.edu/search")
    driver.find_element_by_id("srch").clear()
    driver.find_element_by_id("srch").send_keys("MATH 3323")
    driver.find_element_by_id("srch").send_keys(Keys.RETURN)
    cookies = driver.get_cookies()
    for c in cookies:
        if c["name"] == 'PTGSESSID':
            headers["Cookie"] = 'PTGSESSID=' + c["value"]
    print("new cookie: " + headers["Cookie"])
    driver.close()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

cookie_string="d2d327e693509a987c4d9783b7a260e3f"
# cookie_string="a932166bb57aff99a121259288de5571"
headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'DNT': '1',
    'Host': 'coursebook.utdallas.edu',
    'Origin': 'https://coursebook.utdallas.edu',
    'Referer': 'https://coursebook.utdallas.edu/search',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36 Edg/84.0.522.63',
    'X-Requested-With': 'XMLHttpRequest',
    'Cookie': 'PTGSESSID=' + cookie_string
}

text_file = open("courseCombine.txt", "r")
courses = text_file.read().split("\n")
text_file.close()

@app.route("/change_cookie")
def cookie_change():
    try:
        """cookie_string = request.args.get('cookie').strip()
        f = open(cookie_filename, "w")
        f.write(cookie_string)
        f.close()
        headers = {
          'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
          'Cookie': 'PTGSESSID=' + cookie_string
        }
        return "<p>New cookie " + cookie_string + " successfully stored.</p>"""
        url = 'https://coursebook.utdallas.edu/'
        response = requests.get(url)
        str = response.headers['Set-Cookie']
        cookie_string = re.findall('PTGSESSID=([^;]*)', str)[0]
        return "<p>New cookie " + cookie_string + " successfully stored.</p>"
    except Exception as e:
        return "<h1>Error</h1>"
# cookie_change()

@app.route("/hello")
def home_view():
    return "<h1>Hello World!</h1>"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = 'app/uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

save_dir = '/tmp/' # '/tmp/'

@app.route('/api/convert', methods=['GET', 'POST'])
def convertFile():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist("file")
        flags = request.form.getlist('flags')
        print(request.files)
        imList = []
        if len(files) >= 1:
            for file in files:
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    all_path = os.path.join(save_dir, filename)
                    file.save(all_path)
                    imList.append(all_path)
            processFiles(imList, flags)
            return redirect(url_for('upload_convert',
                                    filename='all.pdf'))
        else:
            return '''
            <!doctype html>
            <title>Pictures to PDF</title>
            <h1>Upload Files</h1>
            <p>At least one file required!</p>
            <form method=post enctype=multipart/form-data>
              <input id=file type=file name=file multiple=>
              <input type=submit value=Upload>
            </form>
            '''             
    return '''
    <!doctype html>
    
    <title>Pictures to PDF</title>
    <h1>Upload Files</h1>
    <form method=post enctype=multipart/form-data>
      <input id=file type=file name=file multiple>
      <label>Same size:
      <input type=checkbox name=flags value=sameSize>
      </label>
      <input type=submit value=Upload>
    </form>
    '''

def processFiles(files, flags):
    print(flags)
    imgs = [image_transpose_exif(rgbaToRgb(file)) for file in files]
    for file in files:
        os.remove(file)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = []
    if 'sameSize' in flags:
        imgs_comb = [i.resize(min_shape) for i in imgs]
    else:
        imgs_comb = imgs

    imgs_comb[0].save( save_dir + 'all.pdf', save_all=True, append_images=imgs_comb[1:])

def image_transpose_exif(im):
    """
    Apply Image.transpose to ensure 0th row of pixels is at the visual
    top of the image, and 0th column is the visual left-hand side.
    Return the original image if unable to determine the orientation.

    As per CIPA DC-008-2012, the orientation field contains an integer,
    1 through 8. Other values are reserved.

    Parameters
    ----------
    im: PIL.Image
       The image to be rotated.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    except Exception:
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)

def rgbaToRgb(filename):
    rgba = Image.open(filename)
    if len(rgba.split()) == 4:
        rgb = Image.new('RGB', rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.split()[3])
        return rgb
    return rgba

@app.route('/api/pdfParse', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(save_dir, filename))
            
            read_pdf = PdfFileReader(save_dir + filename)
            courses = read_pdf.getPage(0).extractText()
            # courses = courses.split("2020 Fall")[1]
            print(courses)
            
            res = re.findall(r"([A-Z]+ [0-9]+)([^\.]*)", courses)
            res = [(a + ' ' + b[:-1].title()) for (a, b) in res]
            resp = jsonify(res)
            return resp
            """return redirect(url_for('uploaded_file',
                                    filename=filename))"""
                                    
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input id=file type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    read_pdf = PdfFileReader(save_dir + filename)
    courses = read_pdf.getPage(0).extractText()
    res = courses.split("2020 Fall")[0].findall(r"[A-Z]+ [0-9]+")
    resp = jsonify(res)
    return resp

@app.route("/icsUpdate", methods=['POST'])
@cross_origin()
def icsUpdate():
    req = request.json
    payload = req['ics']
    ptgid = req['ptg']
    f = open(save_dir + ptgid+".ics", "w")
    f.write(payload)
    f.close()
    response = make_response("Written to file.", 200)
    return response

@app.route("/relay", methods=['POST'])
@cross_origin()
def relay():
    req = request.json
    cookies = req['cookies']
    url = req['url']
    # url = "https://elearning.utdallas.edu/learn/api/public/v1/users/_267278_1/courses"
    payload = {}
    headers = {
        'Connection': 'keep-alive',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Dest': 'document',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cookie': cookies
    }

    response = requests.request("GET", url, headers=headers, data = payload)

    return jsonify(response.json())

@app.route('/ics')
def ics_refresh():
    ptgid = request.args.get('ptg')
    f = open(save_dir + ptgid+".ics", "rb")
    lines = f.read()
    response = make_response(lines, 200)
    response.mimetype = "text/plain"
    return response

@app.route('/converted/<filename>')
def upload_convert(filename):
    upload_path = '../' + save_dir
    if save_dir == '/tmp/':
        upload_path = save_dir
    return send_from_directory(upload_path, filename)

@app.route('/ocr')
def homepage():
    return """
<!DOCTYPE html>
<head>
   <title>My title</title>
   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/kognise/water.css@latest/dist/dark.min.css">
   <style>
        body {
            max-width: 90%;
        }
        #d1 {
            width: 60%;
            float: left;
        }
        #d2 {
            width: 40%;
            float: right;
        }
        #sample {
            max-height: 90vh;
        }
   </style>
   <script>
        function getText() {
            alert("Image submitted; please wait.");
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
                    var tbl = "<table><tr><th>ID</th><th>Word</th><th>Answer</th><th>Confidence</th><th>Correctness (/100)</th></tr>";
                    for (var key in p) {
                        if (p.hasOwnProperty(key)) {
                            tbl += "<tr><td>" + key + "</td><td>" + p[key]["word"] + "</td><td>" + p[key]["answer"] + "</td><td>" + p[key]["confidence"] + "</td><td>" + p[key]["dist"] + "</td></tr>";
                        }
                    }
                    tbl += "</table>";
                    document.getElementById("d2").innerHTML = tbl;
                    console.log(tbl)
                };
                var inp = {"image": reader.result.split(',')[1]};
                xhr.send(JSON.stringify(inp));
            }
            reader.readAsDataURL(file);
        }
   </script>
</head>
<body style="width: 100%; margin: auto;">  
    <div id="d1">
        <h1>Homework OCR</h1>
        <a>
            <img id="sample">
        </a>
        <input type="file" id="img" name="img" accept="image/*" onchange="document.getElementById('sample').src = window.URL.createObjectURL(this.files[0])">
        <button onclick="getText()">Submit</button>
    </div>
    <div id="d2">
        
    </div>
</body>
"""

@app.route('/testOCR')
def testpage():
    return """
<!DOCTYPE html>
<head>
   <title>My title</title>
   <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/kognise/water.css@latest/dist/dark.min.css">
   <script>
        function getText() {
            var query = document.getElementById("srch").value;
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/course', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = function () {
                //  p = JSON.parse(this.responseText);
                document.getElementById("output").innerHTML = this.responseText;
                console.log(this.responseText)
            };
            var inp = {"query": query};
            xhr.send(JSON.stringify(inp));
        }
   </script>
</head>
<body style="width: 880px; margin: auto;">  
    <input id="srch" type="text" placeholder="Search Term" />
    <button onclick="getText()">Submit</button>
    <div id="output"></div>
</body>
"""

url = "http://utdrmp.herokuapp.com/api/rmp?"
# url = "http://localhost:8080/api/rmp?names="

@app.route("/api/smart", methods=['GET'])
@cross_origin()
def smartSearch():
    payload = request.args.get('query').upper()
    filtered = ({"name":a.split(";")[0], "sid":a.split(";")[1]} for a in courses if payload in a.upper())
    result = list(islice(filtered, 10))
    # print(result)
    resp = jsonify(result)
    return resp


def get_query(query):
    print("acquiring html...")
    payload = "action=search&s[]=" + query + "&s[]=term_20f"
    print(payload)
    try:
        conn.request("POST", "/clips/clip-cb11.zog", payload, headers)
    except Exception as e:
        conn = http.client.HTTPSConnection("coursebook.utdallas.edu")
        conn.request("POST", "/clips/clip-cb11.zog", payload, headers)
    # print(conn)
    res = conn.getresponse()
    
    data = res.read().decode("utf-8")
    json_obj = []
    try:
        json_obj = json.loads(data)
    except Exception as e:
        seleniumRefresh()
        print("refreshing...")
        return get_query(query)

    # print(json_obj["sethtml"]["#sr"])
    # print(data)
    html = json_obj["sethtml"]["#sr"]
    s = html.replace("\\n", "\n").replace("\\", "")
    print("acquired.")
    print("collecting...")
    soup = bs4(s, 'html.parser')
    if len(soup.find_all('tbody')) != 1:
        return []
        
    data = []
    totalQuery = url
    i = 0
    for entry in soup.find('tbody').find_all('tr'):
        text = {}
        all_td = entry.find_all('td')
        all_td.pop(2) # CB added hidden element on 8/12/2020, moved to [2] on 9/4/2020
        arry = all_td[1].find('a').text.split('.')
        text["open"] = "Open" if "Open" in all_td[0].text else "Full"
        text["id"] = arry[1]
        text["sid"] = arry[0]
        text["name"] = all_td[2].text;
        text["status"] = all_td[1].find(text=True, recursive=False).strip()
        text["professor"] = all_td[3].text.strip();
        #if(text["professor"] == "Ivor Page"):
            #text["status"] = "Remote/Virtual Learning"
        totalQuery += "names=" + text["professor"] + "&"
        a = all_td[4].findAll(text=True)
        if len(a) >= 4:
            text["time"] = a[0] + '\n' + a[1] + '\n' + a[3]
        else:
            text["time"] = ""
        data.append(text)
    response = requests.request("GET", totalQuery, headers={}, data = {})
    resps = response.text.encode('utf8')
    # print(resps)
    resp_arr = json.loads(resps)
    inx = 0
    for inx in range(0,len(resp_arr)):
        json_array = resp_arr[inx]
        data[inx]["professor_gpa"] = json_array["avgGPA"]
        if json_array["name"] != "N/A":
            data[inx]["professor_rating"] = json_array["rating"]
            data[inx]["professor_link"] = json_array["link"]
        else:
            data[inx]["professor_rating"] = "0 Records Found"
            data[inx]["professor_link"] = "0 Records Found"
    if request.args.get('single') == "true":
        data = data[0]
    # resp = jsonify(data)
    # print("finished with good.")
    # resp.status_code = 200
    return data

def get_queries(queries):
    data = []
    for query in queries:
        data = data + get_query(query.strip())
    resp = jsonify(data)
    print("finished with good.")
    resp.status_code = 200
    return resp

@app.route("/api/coursetest", methods=['GET'])
@cross_origin()
def course_api2():
    return get_queries(request.args.getlist('query'))

my_classes = ["cs1200.hon.20f", "cs3341.hon.20f", "cs3345.hon.20f", "ecs1100.005.20f", "govt2306.006.20f", "math3323.001.20f", "musi3120.501.20f", "univ1010.001.20f"]

@app.route("/api/schedule", methods=['GET'])
@cross_origin()
def schedule():
    return get_queries(my_classes)

@app.route("/api/course", methods=['POST'])
@cross_origin()
def course_api():
    # try:
    req = request.json
    
    print("acquiring html...")
    
    payload = "action=search&s[]=" + req["query"] + "&s[]=term_20f"
    print(payload)
    
    try:
        conn.request("POST", "/clips/clip-cb11.zog", payload, headers)
    except Exception as e:
        conn = http.client.HTTPSConnection("coursebook.utdallas.edu")
        conn.request("POST", "/clips/clip-cb11.zog", payload, headers)
        
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    # print("cookie: " + cookie_string + " data: " + data)
    html = data.split('"#sr":"')[1].split("}}")[0]
    s = html.replace("\\n", "\n").replace("\\", "")
    print("acquired.")
    print("collecting...")
    soup = bs4(s, 'html.parser')
    
    if len(soup.find_all('tbody')) == 1:
        data = []
        for entry in soup.find('tbody').find_all('tr'):
            text = {}
            i = 0
            for i in range(0, len(entry.find_all('td')) - 3):
                text[i] = entry.find_all('td')[i].text
            a = entry.find_all('td')[4].findAll(text=True)
            if len(a) >= 4:
                text[4] = a[0] + '\n' + a[1] + '\n' + a[3]
            else:
                text[4] = ""
            text[5] = entry.find_all('td')[1].find(text=True, recursive=False).strip()
            data.append(text)
        print(len(data))
        resp = jsonify(data)
        print("finished with good.")
    else:
        resp = jsonify([{"bad": "true"}])
        print("finished with bad.")
    resp.status_code = 200
    return resp
    """except Exception as e:
        resp = jsonify([{"bigbad": "true"}])
        print("finished with big bad.")
        resp.status_code = 200
        return resp"""

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
    idx = 1
    if ("analyzeResult" in analysis):
        for line in analysis["analyzeResult"]["readResults"][0]["lines"]:
            seq = difflib.SequenceMatcher(None,key[idx-1],line["text"])
            conf = 0
            for word in line["words"]:
                conf += word["confidence"]
            conf = conf / len(line["words"])
            conf = round(conf, 3)
            dist = str(round(seq.quick_ratio() * 100, 3))
            text[idx] = {"word": line["text"], "confidence":str(conf), "answer":key[idx-1], "dist":dist}
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