import importlib as imt 
import dataset
imt.reload(dataset)


import datetime as dt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

class ModelXception(object):
    '''
    The ModelXception class provides functions to compute bottleneck features,
    train a top (fully-connected) model, and fine tune the model similar
    to example provided in Keras blog.

    Functions:

        get_datagenerators_v1()
            constructs ImageDataGenerator object parameters.
        get_top_model()
            builds the top fully-connected layer
        save_bottlebeck_features()
            feed-forward each sample image once through pretrained model to
            record and save convolution layer output
        train_top_model()
            train top model using saved bottleneck feature
        fine_tune_model()
            fine tune last convolutional block  by training on data set
    '''

    def __init__(self, img_dir= "../data/bin-images/", meta_dir = "../data/metadata/", num_of_class=5, batch_size=32, num_of_data = 1000):
        self.img_dir = img_dir
        self.meta_dir = meta_dir
        self.batch_size = batch_size
        self.num_of_class = num_of_class 
        df = dataset.make_counting_df(img_dir, meta_dir, limit = self.num_of_class, num_of_data = num_of_data)
        self.train_df = df.sample(frac=0.80, random_state=45)
        self.test_df = df.copy()
        self.test_df = self.test_df.drop(self.train_df.index)    
        self.test_label = self.test_df['label'].values

    def get_datagenerators(self, input_size = (299, 299)):
        '''
        Define the image manipulation steps to be randomly applied to each
        image. Multiple versions of this function will likely exist to test
        different strategies.
        Return a generator for both train and test data.
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
            data_format='channels_last',                 # default
            validation_split=0.20)                      # default
        
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
        
        train_generator=train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=self.img_dir,
            x_col="id",
            y_col="label",
            subset="training",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=input_size)

        valid_generator=train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            directory=self.img_dir,
            x_col="id",
            y_col="label",
            subset="validation",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=input_size)

        test_generator=test_datagen.flow_from_dataframe(
            dataframe=self.test_df,
            directory=self.img_dir,
            x_col="id",
            y_col=None,
            batch_size=self.batch_size,
            seed=42,
            shuffle=False,
            class_mode=None,
            target_size=input_size)

        return train_generator, valid_generator, test_generator

    def save_bottlebeck_features(self):

        print('\nComputing train and test bottleneck features ... ...')
        print('\nCreating data generators ... ...')
        print("X_train length = ", self.train_df.size)
        print("X_test length = ", self.test_df.size)
        # data generators are instructions to Keras for further processing of the
        # image data (in batches) before training on the image.
        train_generator, valid_generator, test_generator  = self.get_datagenerators()
        
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
        STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

        print('\nLoading Xception model ... ...')
        model = Xception(include_top=False, weights='imagenet',input_shape=(224, 224, 3))

        # Generate bottleneck data. This is obtained by running the model on the
        # training and test data just once, recording the output (last activation
        # maps before the fully-connected layers) into separate numpy arrays.
        # This data will be used to train and validate a fully-connected model on
        # top of the stored features for computational efficiency.
        print('\nRunning train predictor and saving features ... ...')
        bottleneck_features_train = model.predict(
            generator=train_generator,
            steps=STEP_SIZE_TRAIN,
            max_queue_size=self.batch_size*8,
            #workers=8,
            #use_multiprocessing=False,
            verbose=True)
        np.save('../data/bottleneck_features_train.npy', np.array(bottleneck_features_train))
        print("Train bottleneck feature length = ", len(bottleneck_features_train))


        bottleneck_features_valid = model.predict(
            generator=valid_generator,
            steps=STEP_SIZE_VALID,
            max_queue_size=self.batch_size*8,
            #workers=8,
            #use_multiprocessing=False,
            verbose=True)
        np.save('../data/bottleneck_features_valid.npy', np.array(bottleneck_features_valid))
        print("Validation bottleneck feature length = ", len(bottleneck_features_train))

        print('\nRunning test predictor and saving features ... ...')
        bottleneck_features_test = model.predict(
            generator=test_generator,
            steps=STEP_SIZE_TEST,
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
 
    def run_base_model(self):
        
        input_size = (299,299,3)

        # Prepare Datasets
        train_generator, valid_generator, test_generator  = self.get_datagenerators()
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
        STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

        # Initialize a pretrained model with the Xception architecture and weights pretrained on imagenet
        base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=input_size)

        # ADD TOP
        model = base_model.output
        model = Flatten()(model)
        model = Dense(256, activation='relu')(model)
        model = Dropout(0.5)(model)
        predictions = Dense(self.num_of_class + 1, activation='softmax')(model)
        
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=optimizers.RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        model.fit(
            x=train_generator,
            y=None,
            batch_size=None,
            epochs=10,
            verbose=1,
            callbacks=None,
            validation_data=valid_generator,
            shuffle=False,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_steps=STEP_SIZE_VALID,
            validation_freq=1,
            max_queue_size=10,
            #workers=1,
            #use_multiprocessing=False,
        )

        model.evaluate(
            x=test_generator,
            y=None,
            batch_size=None,
            verbose=1,
            sample_weight=None,
            steps=STEP_SIZE_VALID,
            callbacks=None,
            max_queue_size=10,
            #workers=1,
            #use_multiprocessing=False
        )

    
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

def main():
    # get pre-processed image and label data
    print('\nLoading  ... ...')
    
    model = ModelXception(num_of_class=5)

    start_time = dt.datetime.now()
    #model.save_bottlebeck_features()
    end_time = dt.datetime.now()
    # model.train_top_model()

    # model.fine_tune_model()


if __name__ == '__main__':
    #main()

    model = ModelXception(num_of_class=5)
    model.run_base_model()

    '''
    display_grid = np.zeros((7*2048//16, 16*7))
    for col in range(2048//16):
        for row in range(16): 
            channel_image = X_train[0, :, :, col * 16 + row]
            display_grid[col * 7: (col + 1) * 7, row * 7 : (row + 1) * 7] = channel_image
    '''        