import imutils
import cv2

class AspectAwarePreprocessor:
    """
    CONTRUCTOR
    witdh : desired width
    height : desired height
    inter : interpolation method used when resizing the image
    """
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    """
    image : image to be preprocessed
    """
    def preprocess(self,image):
        # Get wdith and height of image
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if width is the shorter dimension, resize image by width and crop height
        if w < h:
            image = imutils.resize(image, width=self.width,
                                   inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        # if height is the shorter dimension, resize image by height and crop width
        else:
            image = imutils.resize(image, height=self.height,
                               inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # re-grab the width and height and use the deltas to crop the center of the image:
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # our image target image dimensions may be off by ± one pixel; therefore, we make a call to cv2.resize to 
        # ensure our output image has the desired width and height.
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)