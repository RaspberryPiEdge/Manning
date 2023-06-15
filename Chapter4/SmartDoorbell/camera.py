import cv2
import tensorflow as tf
import RPi.GPIO as GPIO
import time
from FaceRecognizer import FaceRecognizer

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)

camera = cv2.VideoCapture(0)
model_path = 'models/face_recognition_quant.tflite'
label_path = 'models/list_of_people.csv'
log_path = 'logs/recognition_timestamps.csv'
cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
delay = 0
face_recognizer = FaceRecognizer(model_path, label_path, log_path)

def blink_led():
    GPIO.output(4, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(4, GPIO.LOW)
    time.sleep(0.5)

while True:
    ret, image = camera.read()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = cascade.detectMultiScale(rgb_image, minNeighbors=20)

    if len(faces):
        print("Found face")
        person = face_recognizer.get_prediction(rgb_image)
        if person is not None:
            blink_led()
            text = f"{person} is at the door"
        else:
            text = "Unknown person at the door"
    else:
        text = "Waiting for visitor"

    cv2.putText(image, text, (25, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Door Cam", image)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

GPIO.cleanup()
camera.release()
cv2.destroyAllWindows()
