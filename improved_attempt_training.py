import json
from pathlib import Path
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# global static variables
dtype_mult = 255.0 # unit8
num_classes = 10
X_shape = (-1, 126, 126, 3)
epoch = 200
batch_size = 128

def check_pre_req():
    if (
        Path('./data/X_train_126_preprocessed.npy').is_file() and
        Path('./data/y_train_126_preprocessed.npy').is_file() and
        Path('./data/X_test_126_preprocessed.npy').is_file() and
        Path('./data/y_test_126_preprocessed.npy').is_file()
    ) == False:
        print('Please complete the execution of encode_images.py first!')
        raise SystemExit

def get_preprocessed_dataset():
    print('Loading Dataset')
    
    X_train = np.load('D:/Projects/CIFAR/data/X_train_126_preprocessed.npy')
    y_train = np.load('D:/Projects/CIFAR/data/y_train_126_preprocessed.npy')
    X_test = np.load('D:/Projects/CIFAR/data/X_test_126_preprocessed.npy')
    y_test = np.load('D:/Projects/CIFAR/data/y_test_126_preprocessed.npy')
    
    return X_train, y_train, X_test, y_test

def generate_optimizer():
    return keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=generate_optimizer(),
                  metrics=['accuracy'])

def generate_model():
    # check if model exists if exists then load model from saved state
    if Path('./models/convnet_model_126.json').is_file():
        print('Loading existing model')
        
        with open('./models/convnet_model_126.json') as file:
            model = keras.models.model_from_json(json.load(file))
            file.close()

        # likewise for model weight, if exists load from saved state
        if Path('./models/convnet_weights_126.h5').is_file():
            model.load_weights('./models/convnet_weights_126.h5')

        compile_model(model)

        return model

    print('Loading new model')
    
    model = Sequential()

    # Conv1 126 126 (128)
    model.add(Conv2D(128, (7, 7), padding='same', input_shape=X_shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (7, 7), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Conv2 63 63 (256)
    model.add(Conv2D(256, (7, 7), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.20))
    model.add(Conv2D(256, (7, 7), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.20))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Conv3 32 32 (512)
    model.add(Conv2D(512, (7, 7), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.30))
    model.add(Conv2D(512, (7, 7), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.30))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Conv4 16 16 (1024)
    model.add(Conv2D(1024, (7, 7), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.40))
    model.add(Conv2D(1024, (7, 7), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.40))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # FC
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # compile has to be done impurely
    compile_model(model)

    with open('./models/convnet_model_126.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
        outfile.close()

    return model
    
def image_generator():
    return ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False
    )

def train(model, X_train, y_train, X_test, y_test):
    print('Training model with data augmentation\n')

    datagen = image_generator()
    datagen.fit(X_train)

    # train each iteration individually to back up current state
    # safety measure against potential crashes
    epoch_count = 0
    while epoch_count < epoch:
        epoch_count += 1
        print('Epoch count:', epoch_count)
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=len(X_train) // batch_size,
                            epochs=1,
                            validation_data=(X_test, y_test))
        print('Epoch {} done, saving model to file\n'.format(epoch_count))
        model.save_weights('./models/convnet_weights_126.h5')

    return model

def get_accuracy(pred, real):
    # reward algorithm
    result = pred.argmax(axis=1) == real.argmax(axis=1)
    return np.sum(result) / len(result)
    
def main():
    print('Welcome to the improved attempt!')
    check_pre_req()
    X_train, y_train, X_test, y_test = get_preprocessed_dataset()
    model = generate_model()
    model = train(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    # execute only if run as a script
    main()

