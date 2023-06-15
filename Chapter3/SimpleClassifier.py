from tflite_support.task.vision import ImageClassifierOptions
from tflite_support.task.vision import ImageClassifier
from tflite_support.task.vision import TensorImage
from tflite_support.task.core import BaseOptions

class SimpleClassifier:
    
    def __init__(self, model):
        options = ImageClassifierOptions( \
			base_options=BaseOptions(file_name=model))

        self.classifier = ImageClassifier.create_from_options \
			(options)
        
    def get_prediction(self, image):
        if type(image) == str:
            tensor_image = \
				TensorImage.create_from_file(image)
        else:
            tensor_image = TensorImage(image)

        result = self.classifier.classify(tensor_image)
        return result.classifications[0].categories[0].category_name, \
			result.classifications[0].categories[0].score
    
if __name__ == '__main__':
    image = 'Banana.png'
    model = 'efficientnet_lite0.tflite'
    simple_classifier = SimpleClassifier(model)
    label, score = simple_classifier.get_prediction(image)
    print(f"Label: {label}	Score: {score}")
