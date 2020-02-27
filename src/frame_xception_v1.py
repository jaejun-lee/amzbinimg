import importlib as imt 
import dataset
imt.reload(dataset)


import datetime as dt
import tensorflow as tf


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import Model


import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

import gc


class framework_baseline(object):
    '''
    This class provides functions to train baseline model based on the 
    previous cohort project from https://github.com/dandresky/inventory-classifier
    .

    Functions:

        load_datagenerators()
            constructs ImageDataGenerator object parameters.
        load_top_model()
            builds the top fully-connected layer
        save_bottlebeck_features()
            feed-forward each sample image once through pretrained model to
            record and save convolution layer output
        run_base_model()
    '''

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def load_datagenerators(self, X_train, y_train, X_test, y_test, input_size = (299, 299)):
        '''
        ImageDataGenerator with Pandas Dataframe id:image file path, label:count as string.
        '''
        train_datagen = ImageDataGenerator(featurewise_center=False, # default
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
            validation_split = 0.2,
            data_format='channels_last'                 # default
            )                      # default
        
        test_datagen = ImageDataGenerator(featurewise_center=False,  # default
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
        
        self.train_generator=train_datagen.flow(
            X_train,
            y_train,    # labels just get passed through
            batch_size=self.batch_size,
            shuffle=True,
            subset = "training",
            seed=None)

        self.valid_generator=train_datagen.flow(
            X_train,
            y_train,    # labels just get passed through
            batch_size=self.batch_size,
            shuffle=True,
            subset = "validation",
            seed=None)

        del X_train
        del y_train
        gc.collect()
        self.STEP_SIZE_TRAIN=self.train_generator.n//self.train_generator.batch_size
        self.STEP_SIZE_VALID=self.valid_generator.n//self.valid_generator.batch_size

        self.test_generator = test_datagen.flow(
            X_test,
            y_test, # labels just get passed through
            batch_size=self.batch_size,
            shuffle=False,
            seed=None)

        del X_test
        del y_test
        gc.collect()

        self.STEP_SIZE_TRAIN=self.train_generator.n//self.train_generator.batch_size
        self.STEP_SIZE_VALID=self.valid_generator.n//self.valid_generator.batch_size
        self.STEP_SIZE_TEST=self.test_generator.n//self.test_generator.batch_size
        
    def save_bottlebeck_features(self):

        print('\nComputing train and test bottleneck features ... ...')
        print('\nCreating data generators ... ...')
        print("X_train length = ", self.train_df.size)
        print("X_test length = ", self.test_df.size)
        # data generators are instructions to Keras for further processing of the
        # image data (in batches) before training on the image.
        train_generator, valid_generator, test_generator  = self.load_datagenerators()
        

        print('\nLoading Xception model ... ...')
        model = Xception(include_top=False, weights='imagenet',input_shape=(299, 299, 3))

        # Generate bottleneck data. This is obtained by running the model on the
        # training and test data just once, recording the output (last activation
        # maps before the fully-connected layers) into separate numpy arrays.
        # This data will be used to train and validate a fully-connected model on
        # top of the stored features for computational efficiency.
        print('\nRunning train predictor and saving features ... ...')
        bottleneck_features_train = model.predict(
            generator=train_generator,
            steps=self.STEP_SIZE_TRAIN,
            max_queue_size=self.batch_size*8,
            #workers=8,
            #use_multiprocessing=False,
            verbose=True)
        np.save('../data/bottleneck_features_train.npy', np.array(bottleneck_features_train))
        print("Train bottleneck feature length = ", len(bottleneck_features_train))


        bottleneck_features_valid = model.predict(
            generator=valid_generator,
            steps=self.STEP_SIZE_VALID,
            max_queue_size=self.batch_size*8,
            #workers=8,
            #use_multiprocessing=False,
            verbose=True)
        np.save('../data/bottleneck_features_valid.npy', np.array(bottleneck_features_valid))
        print("Validation bottleneck feature length = ", len(bottleneck_features_train))

        print('\nRunning test predictor and saving features ... ...')
        bottleneck_features_test = model.predict(
            generator=test_generator,
            steps=self.STEP_SIZE_TEST,
            max_queue_size=self.batch_size*8,
            #workers=8,
            #use_multiprocessing=False,
            verbose=True)
        np.save('../data/bottleneck_features_test.npy',
                np.array(bottleneck_features_test))
        print("Test bottleneck feature length = ", len(bottleneck_features_test))

        # steps_per_epoch=X_train.shape[0] // batch_size in the code above can
        # result in discarding the last few data samples if batch_size doesn't
        # divide evenly into the shape of the data array. Remove the top few samples
        # from the instance variables to keep them equal.

    
    def load_top_model(self, base_model):

        self.top_model = base_model.output
        self.top_model = Dense(256, activation='relu')(self.top_model)
        self.top_model = Dropout(0.5)(self.top_model)
        self.top_model = Dense(1, activation='relu')(self.top_model)

        return self.top_model

    def load_base_model(self, input_shape):
        
        self.base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)


        
    def run_base_model(self):
        pass

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True
        

def process_img(filename):
    """
    Loads image from filename, preprocesses it and expands the dimensions because the model predict function expects a batch of images, not one image
    """
    original = load_img(filename, target_size = (299,299))
    numpy_image = preprocess_input(img_to_array(original))
    image_batch = np.expand_dims(numpy_image, axis =0)

    return image_batch

def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

def main():

    pass

if __name__ == '__main__':


    # get pre-processed image and label data
    print('\nLoading numpy arrays ... ...')
    start_time = dt.datetime.now()
    X_train = np.load('../data/processed_training_images.npy')
    X_test = np.load('../data/processed_test_images.npy')
    y_train = np.load('../data/training_labels.npy')
    y_test = np.load('../data/test_labels.npy')
    stop_time = dt.datetime.now()
    print("Loading arrays took ", (stop_time - start_time).total_seconds(), "s.\n")

    frame = framework_baseline(batch_size=32)

    input_size = (299,299,3)

    # Prepare Datasets
    frame.load_datagenerators(X_train, y_train, X_test, y_test, input_size = (299, 299))  

    # load base model
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(299, 299, 3))

    # load top model
    top_model = base_model.output
    top_model = Flatten()(top_model)
    top_model = Dense(256, activation='relu')(top_model)
    top_model = Dropout(0.5)(top_model)
    predictions = Dense(1, activation='relu')(top_model)

    # stack
    model = Model(inputs= base_model.input, outputs= predictions)
    #print(model.summary())

    # set trainable layer to top layers only
    change_trainable_layers(model, 132)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='mean_squared_error',
                    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                    metrics=['accuracy', 'mae'])

    
    
    print('\nFitting the model ... ...')
    log_dir="../logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        x=frame.train_generator,
        #y=self.train_df['label'].values,
        #generator = train_generator,
        batch_size=None,
        epochs=1,
        verbose=1,
        validation_data=frame.valid_generator,
        #shuffle=False,
        #class_weight=None,
        #sample_weight=None,
        #initial_epoch=0,
        steps_per_epoch=frame.STEP_SIZE_TRAIN,
        validation_steps=frame.STEP_SIZE_VALID,
        #validation_freq=1,
        max_queue_size=frame.batch_size*8,
        #workers=4,
        use_multiprocessing=False,
        callbacks=[tensorboard_callback]
    )

    print('\nValidationg the model ... ...')
    log_dir="../logs/validation/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.evaluate(
            x=frame.test_generator,
            y=None,
            batch_size=None,
            verbose=1,
            sample_weight=None,
            steps=frame.STEP_SIZE_TEST,
            max_queue_size=frame.batch_size*8,
            #workers=1,
            use_multiprocessing=False,
            callbacks=[tensorboard_callback]
        )

    pred = model.predict(x=frame.test_generator,
            steps=frame.STEP_SIZE_TEST,
            max_queue_size=frame.batch_size*8,
            #workers=8,
            use_multiprocessing=False,
            verbose=True)
    
    '''
    display_grid = np.zeros((7*2048//16, 16*7))
    for col in range(2048//16):
        for row in range(16): 
            channel_image = X_train[0, :, :, col * 16 + row]
            display_grid[col * 7: (col + 1) * 7, row * 7 : (row + 1) * 7] = channel_image
    '''        