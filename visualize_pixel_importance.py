import sys
import json
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

model_path = './models/stashed/convnet_model.json'
weight_path = './models/stashed/convnet_weights.h5'
dtype_mult = 255
num_classes = 10
X_shape = (-1,32,32,3)
layer_depths = ['conv2d_1','conv2d_2','conv2d_3','conv2d_4','conv2d_5','conv2d_6']

labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def get_dataset():
    sys.stdout.write('Loading Dataset\n\n')
    sys.stdout.flush()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # we perform a series of normalization and binarizer on the dataset here
    X_train = X_train.astype('float32') / dtype_mult
    X_test = X_test.astype('float32') / dtype_mult
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

def load_model():
    if (Path('./models/convnet_improved_model.json').is_file() == False) | (Path('./models/convnet_improved_model.json').is_file() == False):
        sys.stdout.write('Please train model using basic_model.py first')
        sys.stdout.flush()
        #raise SystemExit

    with open(model_path) as file:
        model = keras.models.model_from_json(json.load(file))
        file.close()

    model.load_weights(weight_path)

    return model

def get_random_img(X, y):
    i = np.random.randint(0, len(X))
    img = X[i].reshape(X_shape)
    label_id = y[i].argmax()

    return img, label_id

def get_random_correct_img(X, y, model):
    found = False

    while found == False:
        img, label_id = get_random_img(X, y)
        pred = model.predict(img)[0]
        if pred.argmax() == label_id:
            found = True

    return img, label_id

def pixel_contribution(img, label_id, i, j, model, certainty):
    img.squeeze()[i:i+5,j:j+5] = np.random.rand(5,5,X_shape[3])
    pred = model.predict(img)
    contribution = certainty - pred.squeeze()[label_id]

    return contribution

def visualize(X, y, model, n_imgs=3):
    sys.stdout.write('Regions with higher importance for model accuracy are shaded in red, less important regions are shaded in yellow\n')
    sys.stdout.flush()

    k=0
    while k < n_imgs:
        img, label_id = get_random_correct_img(X, y, model)
        pixel_contribution_img = np.zeros((X_shape[1:3]))

        pred = model.predict(img)
        certainty = pred.squeeze()[label_id]

        for i in range(X_shape[1]-4):
            for j in range(X_shape[2]-4):
                pixel_contribution_img[i+2,j+2] = pixel_contribution(
                    img.copy(), label_id,
                    i, j, model, certainty)

        if pixel_contribution_img.max() > 0.25:
            _ = plt.imshow(img.squeeze(), alpha=0.9)
            _ = plt.imshow(pixel_contribution_img, cmap='YlOrRd', alpha=0.3)
            _ = plt.title(labels[label_id])
            _ = plt.show()
            k += 1

def main():
    _, _, X, y = get_dataset()
    model = load_model()
    visualize(X, y, model, n_imgs=3)

if __name__ == "__main__":
    # execute only if run as a script
    main()
