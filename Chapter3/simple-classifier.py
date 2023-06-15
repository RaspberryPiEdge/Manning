from tflite_support.task.vision import ImageClassifierOptions
from tflite_support.task.vision import ImageClassifier
from tflite_support.task.vision import TensorImage
from tflite_support.task.core import BaseOptions

options = ImageClassifierOptions(base_options=BaseOptions(file_name='efficientnet_lite0.tflite'))
classifier = ImageClassifier.create_from_options(options)

image = TensorImage.create_from_file('Banana.png')

classification_result = classifier.classify(image)

print(classification_result.classifications[0].categories[0].category_name)
print(classification_result.classifications[0].categories[0].score)

