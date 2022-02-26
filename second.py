from flask import Flask,render_template,Response
import numpy as np
from cv2 import cv2

app=Flask(__name__)
camera=cv2.VideoCapture(0)
nose_cascade = cv2.CascadeClassifier(r'C:\Users\lenovo\Downloads\Tutorial%207\Tutorial%207\haarcascade_nose.xml')
img = cv2.imread('dognose.png')



def generate_frames():
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            nose = nose_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in nose:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 0)
                resized_img = cv2.resize(img, (w, h))
                frame[ y:y+h , x:x+w ] = resized_img

            cv2.imshow('frame', frame)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

