import tensorflow as tf
import numpy as np
from PIL import Image
import time

class TFLiteClassifier:

    def __init__(self, model_path='models/lite-model_mobilenet_v2_100_224_fp32_1.tflite'):
        # Load TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image):
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = Image.open(image)
            
        #image = Image.open(image).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = (image_array / 127.5) - 1  # Normalize to [-1, 1]
        image_array = np.expand_dims(image_array, axis=0)
        return image_array.astype('float32')

    def get_prediction(self, image):
        image_array = self.preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)

        start_time = time.time()
        self.interpreter.invoke()
        end_time = time.time()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output_data)

        elapsed_time = round(end_time - start_time, 4)

        return predicted_class, elapsed_time


if __name__ == "__main__":
    classifier = TFLiteClassifier()
    predicted_class, inference_time = classifier.get_prediction('images/Banana.png')

    BANANA_CLASS_INDEX = 955
    if predicted_class == BANANA_CLASS_INDEX:
        print("Yes, it's a banana!")
    else:
        print("No, it's not a banana!")
    print(f"Inference time: {inference_time:.4f} seconds")
