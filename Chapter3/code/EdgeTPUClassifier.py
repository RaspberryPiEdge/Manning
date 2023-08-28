import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
import time

class EdgeTPUClassifier:

    def __init__(self, model_path='models/mobilenet_v2_1.0_224_quantized_1_default_1.tflite'):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = Image.open(image)

        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array.astype('uint8')

    def get_prediction(self, image):
        image_array = self.preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)

        start_time = time.time()
        self.interpreter.invoke()
        end_time = time.time()

        classes = classify.get_classes(self.interpreter, top_k=1)
        predicted_class = classes[0].id
        
        elapsed_time = round(end_time - start_time, 4)
        
        return predicted_class, elapsed_time

if __name__ == "__main__":
    classifier = EdgeTPUClassifier()
    predicted_class, inference_time = classifier.get_prediction('images/Banana.png')

    BANANA_CLASS_INDEX = 955
    if predicted_class == BANANA_CLASS_INDEX:
        print("Yes, it's a banana!")
    else:
        print("No, it's not a banana!")
    print(f"Inference time: {inference_time:.4f} seconds")
