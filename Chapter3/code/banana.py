import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='models/mobilenetv2.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
image_path = 'images/Banana.png'
image = Image.open(image_path).convert('RGB')
image = image.resize((224, 224))
image_array = np.expand_dims(np.array(image), axis=0)
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

# Set the image as input tensor
interpreter.set_tensor(input_details[0]['index'], image_array)

# Run the inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get the predicted class
predicted_class = np.argmax(output_data)

# Class index for "banana" in ImageNet
BANANA_CLASS_INDEX = 954

# Check if the prediction corresponds to a banana
if predicted_class == BANANA_CLASS_INDEX:
    print("This is a banana!")
else:
    print("This is not a banana.")
