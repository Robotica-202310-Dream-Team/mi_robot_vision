#!usr/bin/python3

from flask import Flask
from flask import render_template_string
from flask import Response
import cv2

### SI NO FUNCIONA:
### 1. asegurar que boot/config.txt tenga la linea > start_x=1
### 2. 	correr > lsof /dev/video0
###	y luego > kill [PID]
### 	probablemente hay otro programa utilizando la camara por otra razon

### TODO: agregar algo que mate procesos usando camara a boot script

app = Flask("camera_streaming")

cap_micro = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L) #cv2.CAP_DSHOW)

def template_string_generator(video_feed_filename, video_feed_name):
     s = """
     <!DOCTYPE html>
     <html lang="en">
     <head>
          <meta charset="UTF-8">
          <meta http-equiv="X-UA-Compatible" content="IE=edge">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Videostreaming</title>
          <style>
               .container{
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    background-color: #000000;
                    color: black;
                    text-align: center;
               }
          </style>
     </head>
     <body class = "container">
          <h1 style="color:white;">""" + video_feed_name + """</h1>
          <img src="{{ url_for('""" + video_feed_filename + """') }}">
     </body>
     </html>
     """
     print(s)
     return s

def generate_camera_streaming():
     while True:
          ret2, frame2 = cap_micro.read()
          if ret2:
               (flag2, encodedImage2) = cv2.imencode(".jpg", frame2)
               if not flag2:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage2) + b'\r\n')


@app.route("/camera_streaming")
def camera_streaming():
     return render_template_string(template_string_generator("video_feed_camera_streaming", "Camara camera_streaming"))


@app.route("/video_feed_camera_streaming")
def video_feed_camera_streaming():
     return Response(generate_camera_streaming(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
     app.run(debug=False, port=3000, host="0.0.0.0")

cap_micro.release()