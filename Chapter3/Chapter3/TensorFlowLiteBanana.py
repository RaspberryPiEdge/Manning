import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='models/lite-model_mobilenet_v2_100_224_fp32_1.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = (image_array / 127.5) - 1  # Normalize to [-1, 1]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype('float32')


image_path = 'images/Banana.png'
image_array = preprocess_image(image_path)

# Set the image as input tensor
interpreter.set_tensor(input_details[0]['index'], image_array)

# Start timer
start_time = time.time()

# Run the inference
interpreter.invoke()

# End timer and print inference time
end_time = time.time()
print(f"Inference time: {end_time - start_time:.4f} seconds")

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get the predicted class
predicted_class = np.argmax(output_data)

# Class index for "banana" in ImageNet
BANANA_CLASS_INDEX = 955

# Check if the prediction corresponds to a banana
if predicted_class == BANANA_CLASS_INDEX:
    print("Yes, it's a banana!")
else:
    print("No, it's not a banana!")
