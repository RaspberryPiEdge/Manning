import numpy as np
from PIL import Image
import time
from pycoral.adapters import classify
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class TFLiteClassifier:

    def __init__(self, model_path='models/mobilenet_v2_1.0_224_quantized_1_default_1.tflite', labels_path='models/labels.txt'):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(labels_path)

    def preprocess_image(self, image):
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = Image.open(image)
            
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = (image_array / 127.5) - 1  # Normalize to [-1, 1]
        image_array = np.expand_dims(image_array, axis=0)
        return image_array.astype('float32')

    def get_prediction(self, image):
        image_array = self.preprocess_image(image)
        
        start_time = time.time()
        self.interpreter.set_tensor(self.interpreter.input_details[0]['index'], image_array)
        self.interpreter.invoke()
        classes = classify.get_classes(self.interpreter, top_k=1)
        end_time = time.time()

        elapsed_time = round(end_time - start_time, 2)

        return classes[0].id, self.labels[classes[0].id], elapsed_time

if __name__ == "__main__":
    classifier = TFLiteClassifier()
    predicted_class, label_name, inference_time = classifier.get_prediction('images/Banana.png')

    if "banana" in label_name.lower():
        print("Yes, it's a banana!")
    else:
        print("No, it's not a banana!")
    print(f"Inference time: {inference_time:.4f} seconds")
