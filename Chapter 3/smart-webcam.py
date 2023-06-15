import cv2
import tensorflow as tf
from SimpleClassifier import SimpleClassifier

camera = cv2.VideoCapture(0)
simple_classifier = SimpleClassifier('efficientnet_lite0.tflite')

while True:    
    ret, image = camera.read()
    resized_image = cv2.resize(image, (224, 224)) 
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    label, score = simple_classifier.get_prediction(resized_image)
    text = f"Label: {label} Score: {score}"
    
    cv2.putText(image,text,(25,450),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
    cv2.imshow("Video Feed", image)

    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
