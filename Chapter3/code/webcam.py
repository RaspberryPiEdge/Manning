import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    cv2.imshow("Video Feed", image)

    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
