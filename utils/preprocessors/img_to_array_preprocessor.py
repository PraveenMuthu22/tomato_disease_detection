from keras.preprocessing.image import img_to_array


class ImgToArrayPreprocessor:
    def __init__(self, format_of_data=None):
        # Store the image data format
        self.format_of_data = format_of_data

    def preprocess(self, image):
        return img_to_array(image, data_format=self.format_of_data)
