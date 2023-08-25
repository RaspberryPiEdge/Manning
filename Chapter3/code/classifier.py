import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load TFLite model
model_path = 'models/mobilenetv2.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
image_path = 'images/casa-loma.png' # Change this to the path of your image
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

# Load the labels
labels_path = 'models/imagenet_labels.txt'
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Get the label corresponding to the predicted class
predicted_label = labels[predicted_class + 1] # +1 as the first label is usually background
print(f'The predicted label for the image is: {predicted_label}')
