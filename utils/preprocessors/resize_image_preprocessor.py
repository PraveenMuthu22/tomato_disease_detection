import imutils
import cv2

class resizeImagePreprocessor:
    """
    CONTRUCTOR
    witdh : desired width
    height : desired height
    interpolation : for resizing image
    """
    def __init__(self,width,height,interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    """
    image : image to be preprocessed
    """
    def preprocess(self,image):
        # Get wdith and height of image
        (h, w) = image.shape[:2]
        dWeight = 0
        dHeight = 0

        # if width shorter, resize based on width and crop height
        if w < h:
            image = imutils.resize(image, width=self.width,
                                   inter=self.interpolation)
            dHeight = int((image.shape[0] - self.height) / 2.0)

        # if height shorter, resize based on height and crop width
        else:
            image = imutils.resize(image, height=self.height,
                               inter=self.interpolation)
            dWeight = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dHeight:h - dHeight, dWeight:w - dWeight]

        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.interpolation)