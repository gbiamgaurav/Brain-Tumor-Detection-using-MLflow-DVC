
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class braindata:
    def __init__(self,filename):
        self.filename =filename

    def predictbraindata(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'giloma: Tumor'
            return [{ "image" : prediction}]
        elif result[0] == 1:
            prediction = 'meningioma: Tumor'
            return [{ "image" : prediction}]
        elif result[0] == 2:
            prediction = 'notumor : No Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'pituitary: Tumor'
            return [{ "image" : prediction}]