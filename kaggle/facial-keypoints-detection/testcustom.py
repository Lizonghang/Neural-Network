from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model


def load_image(filepath):
    im = Image.open(filepath)
    im = im.convert('L')
    im = im.resize((96, 96))
    return np.array(im).reshape((1, 96, 96, 1)) / 255.0


def display(X, y_pred):
    plt.figure()
    plt.imshow(X.reshape((96, 96)), cmap='gray')
    plt.axis('off')
    y_pred = y_pred.clip(0, 1)
    plt.scatter(y_pred[0::2] * 96.0, y_pred[1::2] * 96.0, c='r', marker='x')
    plt.show()


if __name__ == '__main__':
    model = load_model('ckpt/model.h5')
    X = load_image('test1.png')
    y_pred = model.predict(X).reshape((30,))
    display(X, y_pred)
