import datetime
import numpy as np
from PIL import Image
from LogRecord import LogRecord
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class FaceRecognizer:
    def __init__(self, model_path, label_path, log_path, threshold=0.9):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(label_path)
        self.log_record = LogRecord(log_path)
        self.threshold = threshold
        self.last_prediction = None
        self.last_prediction_time = datetime.datetime.min

    def get_prediction(self, image):
        # Load image and preprocess it
        input_image = self.preprocess_image(image)
        common.set_input(self.interpreter, input_image)

        # Run the interpreter
        self.interpreter.invoke()

        # Get the classification results
        classes = classify.get_classes(self.interpreter, top_k=1)
        
        if classes and classes[0].score < self.threshold:
            return None

        # Determine the most probable label
        predicted_label = classes[0].id if classes else None
        confidence = classes[0].score if classes else None

        # Record the timestamp and label
        current_time = datetime.datetime.now()

        # Check if the same label has been predicted within the last 10 seconds
        if (
            self.last_prediction == self.labels[predicted_label]
            and (current_time - self.last_prediction_time).total_seconds() < 10
        ):
            return self.labels[predicted_label]

        self.last_prediction = self.labels[predicted_label]
        self.last_prediction_time = current_time

        self.log_record.record_timestamp(self.labels[predicted_label])

        return self.labels[predicted_label]

    @staticmethod
    def preprocess_image(image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = Image.open(image)

        image = image.convert('RGB')
        image = image.resize((224, 224), Image.BICUBIC)
        return np.array(image, dtype=np.uint8)  # Corrected dtype back to uint8


if __name__ == "__main__":
    model_path = 'models/face_recognition_quant_edgetpu.tflite'
    label_path = 'models/list_of_people.csv'
    log_path = 'logs/recognition_timestamps.csv'
    image_path = 'images/logan.png'

    face_recognizer = FaceRecognizer(model_path, label_path, log_path)
    predicted_label = face_recognizer.get_prediction(image_path)

    if predicted_label is not None:
        print("The person in the picture is:", predicted_label)
    else:
        print("The person in the picture is not recognized.")
