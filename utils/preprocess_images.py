import numpy as np
import os

from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def preprocess_images(images_dir, image_size=50):
    X, Y = [], []
    class_index = {}
    classes = os.listdir(images_dir)
    for index, name in enumerate(classes):
        class_index[index] = name
        files = os.listdir(os.path.join(images_dir,name))
        for file in files:
            image = Image.open(os.path.join(images_dir,name,file))
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X.append(data)
            Y.append(index)

    X = np.array(X)
    Y = np.array(Y)

    X = X.astype("float32")
    X /= 255.0
    Y = np_utils.to_categorical(Y, len(classes))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    return X_train, X_test, y_train, y_test
