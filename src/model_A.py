import datetime as dt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.utils.np_utils import to_categorical
import logging
import numpy as np
import os
import pickle
import random
import tensorflow as tf

BATCH_SIZE = 64
EPOCHS = 40
# Set NUM_CLASSES to 0 to look for empty bins. Set it to Qty+1 to count items
NUM_CLASSES = 6
OPTIMIZER = 'sgd'


def build_model(input_shape):

    logging.info("Build Convolutional and Fully Connected layers ...")
    print("Build Convolutional and Fully Connected layers ... ")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if NUM_CLASSES == 0:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])
    else:
        model.add(Dense(NUM_CLASSES))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])

    return model

def build_vgg17_model(input_shape):

    logging.info("Build Convolutional and Fully Connected layers ...")
    print("Build Convolutional and Fully Connected layers ... ")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    if NUM_CLASSES == 0:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])
    else:
        model.add(Dense(NUM_CLASSES))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])
    return model

def get_datagenerators_v1(X_train, y_train, X_test, y_test):
    '''
    Define the image manipulation steps to be randomly applied to each image.
    Multiple versions of this function will likely exist to test different
    strategies. Return a generator for both train and test data.
    '''
    print("Create Image Data Generators for train and test ... ")
    train_datagen = idg(featurewise_center=False, # default
        samplewise_center=False,                    # default
        featurewise_std_normalization=False,        # default
        samplewise_std_normalization=False,         # default
        zca_whitening=False,                        # default
        zca_epsilon=1e-6,                           # default
        rotation_range=0.,                          # default
        width_shift_range=0.,                       # default
        height_shift_range=0.,                      # default
        shear_range=0.,                             # default
        zoom_range=0.,                              # default
        channel_shift_range=0.,                     # default
        fill_mode='nearest',                        # default
        cval=0.,                                    # default
        horizontal_flip=False,                      # default
        vertical_flip=False,                        # default
        rescale=1./255,                             # rescale RGB vales
        preprocessing_function=None,                # default
        data_format='channels_last')                # default
    test_datagen = idg(featurewise_center=False,  # default
        samplewise_center=False,                    # default
        featurewise_std_normalization=False,        # default
        samplewise_std_normalization=False,         # default
        zca_whitening=False,                        # default
        zca_epsilon=1e-6,                           # default
        rotation_range=0.,                          # default
        width_shift_range=0.,                       # default
        height_shift_range=0.,                      # default
        shear_range=0.,                             # default
        zoom_range=0.,                              # default
        channel_shift_range=0.,                     # default
        fill_mode='nearest',                        # default
        cval=0.,                                    # default
        horizontal_flip=False,                      # default
        vertical_flip=False,                        # default
        rescale=1./255,                             # rescale RGB vales
        preprocessing_function=None,                # default
        data_format='channels_last')                # default

    train_generator = train_datagen.flow(
        X_train,
        y_train,    # labels just get passed through
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=None)
    test_generator = test_datagen.flow(
        X_test,
        y_test, # labels just get passed through
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=None)

    return train_generator, test_generator

def get_data():
    '''
    Images have already been screened, resized, and converted to numpy arrays.
    They are are stored in ../../dsi-capstone-data/
        processed_training_images.npy
        processed_test_images.npy
        training_labels.npy
        test_labels.npy
    The selected batch size will not always divide evenly into the total number
    of samples which causes errors with the Keras functions that use the image
    data generator. The leftover samples are trimmed from the data set to avoid
    this.
    '''
    logging.info('Loading numpy arrays ...')
    print('\nLoading numpy arrays ... ...')
    X_train = np.load('../../dsi-capstone-data/processed_training_images.npy')
    X_test = np.load('../../dsi-capstone-data/processed_test_images.npy')
    y_train = np.load('../../dsi-capstone-data/training_labels.npy')
    y_test = np.load('../../dsi-capstone-data/test_labels.npy')

    logging.info('Trimming data to integer number of batches ...')
    print("Trimming data to integer number of batches ...")
    num_train_batches = X_train // BATCH_SIZE
    num_test_batches = X_test // BATCH_SIZE
    X_train = X_train[:len(num_train_batches * BATCH_SIZE)]
    y_train = y_train[:len(num_train_batches * BATCH_SIZE)]
    X_test = X_test[:len(num_test_batches * BATCH_SIZE)]
    y_test = y_test[:len(num_test_batches * BATCH_SIZE)]
    logging.info("  X_train samples = %d" % X_train.shape[0])
    logging.info("  y_train samples = %d" % y_train.shape[0])
    logging.info("  X_test samples = %d" % X_test.shape[0])
    logging.info("  y_test samples = %d" % y_test.shape[0])

    return X_train, y_train, X_test, y_test

def main():

    # init random seeds to ensure consistent results during evaluation
    random.seed(39)
    np.random.seed(39)
    tf.set_random_seed(39)

    # begin a logging function to record events
    try:
        os.remove('model_A.log')    # delete the existing file to start new
    except OSError:
        pass
    logging.basicConfig(filename='model_A.log',level=logging.DEBUG)
    logging.info('Begin training model A ...')
    logging.info("  Batch size = %s" % BATCH_SIZE)
    logging.info("  Epochs = %s" % EPOCHS)
    logging.info("  Number of classes = %s" % NUM_CLASSES)
    logging.info("  Optimizer = %s" % OPTIMIZER)
    start_time = dt.datetime.now()

    # read pre-processed data and trim to integer number of batches
    X_train, y_train, X_test, y_test = get_data()

    # convert classes to categorical if trying to count items in a bin
    if NUM_CLASSES > 0:
        print(y_train)
        y_train = to_categorical(y_train, NUM_CLASSES)
        print(y_train)
        y_test = to_categorical(y_test, NUM_CLASSES)

    # data generators are instructions to Keras for further processing of the
    # image data (in batches) before training on the image.
    train_generator, test_generator = \
        get_datagenerators_v1(X_train, y_train, X_test, y_test)

    # get the model
    #model = build_model(input_shape=X_train.shape[1:])
    model = build_vgg17_model(input_shape=X_train.shape[1:])
    logging.info("Model Summary \n%s" % model.to_json())
    model.summary()

    logging.info('Fitting the model ...')
    print('\nFitting the model ...')
    hist = model.fit_generator(train_generator,
        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=True,
        callbacks=None,
        validation_data=test_generator,
        validation_steps=X_test.shape[0] // BATCH_SIZE,
        class_weight=None,
        max_queue_size=BATCH_SIZE*8,
        workers=8,
        use_multiprocessing=False,
        initial_epoch=0)

    logging.info('Scoring the model ...')
    print('\nScoring the model ...')
    scores = model.evaluate_generator(test_generator,
        steps=X_test.shape[0] // BATCH_SIZE,
        max_queue_size=BATCH_SIZE*8,
        workers=8,
        use_multiprocessing=False)
    print(scores)
    logging.info("  Loss: %.3f    Accuracy: %.3f" % (scores[0], scores[1]))

    logging.info('Making predictions ...')
    print('\nMaking predictions ...')
    pred = model.predict_generator(test_generator,
        steps=X_test.shape[0] // BATCH_SIZE,
        max_queue_size=BATCH_SIZE*8,
        workers=8,
        use_multiprocessing=False,
        verbose=True)

    print("Predictions: ", pred)
    print("Actual: ", y_test)
    np.save('../../dsi-capstone-data/model_A_predictions.npy', pred)

    logging.info("Saving model ...")
    print("Saving model ...")
    model.save('../../dsi-capstone-data/model_A.h5')

    stop_time = dt.datetime.now()
    print("Scanning and shuffling took ", (stop_time - start_time).total_seconds(), "s.\n")
    logging.info("Training complete. Elapsed time = %.0fs", (stop_time - start_time).total_seconds())

    pickle.dump(hist.history, open("../../dsi-capstone-data/model_A_history.pkl", "wb"))









if __name__ == '__main__':
    main()
