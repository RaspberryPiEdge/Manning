import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import time  # Import the time module

# Load Pre-trained Model
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=(224, 224, 3))
])

# Preprocess the Image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

image_path = "images/Banana.png"
image = preprocess_image(image_path)

# Start timer
start_time = time.time()

# Predict the Class
predictions = model.predict(image)

# End timer and print inference time
end_time = time.time()
print(f"Inference time: {end_time - start_time:.4f} seconds")

predicted_class = tf.argmax(predictions[0], axis=-1).numpy()

# Check if It's a Banana
BANANA_CLASS_INDEX = 955  # Use 955 for TensorFlow model
if predicted_class == BANANA_CLASS_INDEX:
    print("Yes, it's a banana!")
else:
    print("No, it's not a banana!")
