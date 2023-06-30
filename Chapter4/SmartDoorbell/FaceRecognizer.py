import datetime
import numpy as np
from PIL import Image
from LogRecord import LogRecord
import tflite_runtime.interpreter as tflite

class FaceRecognizer:
    def __init__(self, model_path, label_path, log_path, threshold=0.9):
        self.interpreter = self.load_model(model_path)
        self.labels = self.load_labels(label_path)
        self.log_record = LogRecord(log_path)
        self.threshold = threshold
        self.last_prediction = None
        self.last_prediction_time = datetime.datetime.min

    def get_prediction(self, image):
        # Load image and preprocess it
        input_image = self.preprocess_image(image)

        # Get input and output tensors
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Point the data to be used for testing and run the interpreter
        self.interpreter.set_tensor(input_details[0]['index'], input_image)
        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        # Determine the most probable label
        predicted_label = np.argmax(output_data[0])
        confidence = output_data[0][predicted_label]

        if confidence < self.threshold:
            return None

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

    def load_model(self, model_path):
        interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        interpreter.allocate_tensors()
        return interpreter

    def load_labels(self, label_path):
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def preprocess_image(image):
        if isinstance(image, np.ndarray):
            # If the image is a NumPy array (i.e., already an image), convert it to PIL format
            image = Image.fromarray(image)
        else:
            # If the image is a file path, open the image file
            image = Image.open(image)

        image = image.convert('RGB')
        image = image.resize((224, 224), Image.BICUBIC)
        image_array = np.array(image, dtype=np.int8)  # Corrected dtype from uint8 to int8
        image_array = np.expand_dims(image_array, axis=0)
        return image_array


if __name__ == "__main__":
    
    model_path = 'models/face_recognition_quant_edgetpu.tflite'
    label_path = 'models/list_of_people.csv'
    log_path = 'logs/recognition_timestamps.csv'
    image_path = 'images/readers-test-picture.png'

    face_recognizer = FaceRecognizer(model_path, label_path, log_path)
    predicted_label = face_recognizer.get_prediction(image_path)
    
    if predicted_label is not None:
        print("The person in the picture is:", predicted_label)
    else:
        print("The person in the picture is not recognized.")
