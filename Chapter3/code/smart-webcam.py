import cv2
#from TFLiteClassifier import TFLiteClassifier
from ImageNetLabel import ImageNetLabel
from EdgeTPUClassifier import EdgeTPUClassifier

camera = cv2.VideoCapture(0)
#classifier = TFLiteClassifier()
classifier = EdgeTPUClassifier()
label_translator = ImageNetLabel()

while True:    
    ret, image = camera.read()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction_index, inference_time = classifier.get_prediction(rgb_image)
    label = label_translator.get_label(prediction_index)
    text = f"Label: {label} | Inference time: {inference_time}"
    
    cv2.putText(image,text,(25,450),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.imshow("Video Feed", image)

    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
