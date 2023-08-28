class ImageNetLabel:
    def __init__(self, file_path='models/ImageNetLabels.txt'):
        with open(file_path, 'r') as file:
            self.labels = [line.strip() for line in file]
        
    def get_label(self, index):
        return self.labels[index] if index < len(self.labels) else "Unknown"

if __name__ == "__main__":
    labeler = ImageNetLabel()
    BANANA_CLASS_INDEX = 955

    banana_label = labeler.get_label(BANANA_CLASS_INDEX)
    print(f"The label for class index {BANANA_CLASS_INDEX} is: {banana_label}")
