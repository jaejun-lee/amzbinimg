import importlib as imt 
import datasets_v0
imt.reload(datasets_v0)

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
import tensorflow.keras.backend as K

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

import gc


class framework_xception(object):
    '''
    This class provides functions to train baseline model.

    Functions:

        load_datagenerators()
            constructs ImageDataGenerator object parameters.
        load_top_model()
            builds the top fully-connected layer
        save_bottlebeck_features()
            feed-forward each sample image once through pretrained model to
            record and save convolution layer output
        run_base_model()

    TODO:
        1. move loading datasets logic to datasets_v3 module
        2. move plot functions to utils module
        3. move custom loss function to utils module
        4. adapt to use model catalog 

    '''

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def load_datagenerators(self, X_train, y_train, X_test, y_test, input_size = (128, 128)):
        '''set datagen and flow

        NOTE:
            1. first, attempted to use flow from dataframe, but it's raw option does not work
            with integer label correctly.

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

        self.STEP_SIZE_TRAIN=self.train_generator.n//self.train_generator.batch_size
        self.STEP_SIZE_VALID=self.valid_generator.n//self.valid_generator.batch_size

        self.test_generator = test_datagen.flow(
            X_test,
            y_test, # labels just get passed through
            batch_size=self.batch_size,
            shuffle=False,
            seed=None)

        self.STEP_SIZE_TRAIN=self.train_generator.n//self.train_generator.batch_size
        self.STEP_SIZE_VALID=self.valid_generator.n//self.valid_generator.batch_size
        self.STEP_SIZE_TEST=self.test_generator.n//self.test_generator.batch_size

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

def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

def plot_history(history):    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def soft_acc(y_true, y_pred):
    '''caculate accuracy for RMSE loss function prediction.
    '''
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def main():

    pass

def run_prediction():
    '''Procedure to run prediction model in jupyter notebook.
    '''
    # load dataset
    X_train, X_test, y_train, y_test = datasets_v0.load_data()

    frame = framework_xception(batch_size=32)

    input_size = (128,128,3)

    # Prepare Datasets
    frame.load_datagenerators(X_train, y_train, X_test, y_test, input_size = (128, 128))

    # load base model
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=(128, 128, 3))

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

    #optimizer = optimizers.SGD(lr=1e-4, momentum=0.9)
    optimizer = optimizers.Adam(
    learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
    )

    model.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=[soft_acc, tf.keras.metrics.RootMeanSquaredError()])

    print('\nFitting the model ... ...')
    log_dir="../logs/fit/xception/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        x=frame.train_generator,
        #y=self.train_df['label'].values,
        #generator = train_generator,
        batch_size=None,
        epochs=20,
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


if __name__ == '__main__':
    pass

    
    