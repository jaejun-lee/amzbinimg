import datetime as dt
import keras
from keras import applications
from keras import losses
from keras import metrics
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator as idg
import numpy as np
from sklearn.metrics import mean_squared_error

class ModelVGG16(object):
    '''
    The ModelVGG16 class provides functions to compute bottleneck features,
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

    def __init__(self, X_train, y_train, X_test, y_test, batch_size=32):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        pass

    def get_datagenerators_v1(self):
        '''
        Define the image manipulation steps to be randomly applied to each
        image. Multiple versions of this function will likely exist to test
        different strategies.
        Return a generator for both train and test data.
        '''
        train_generator = idg(featurewise_center=False, # default
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
        test_generator = idg(featurewise_center=False,  # default
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
        return train_generator, test_generator

    def get_top_model(self, input_shape):

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu'))

        opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer=opt,
                      loss=losses.mean_squared_error,
                      metrics=[metrics.mae])

        return model

    def save_bottlebeck_features(self):

        print('\nComputing train and test bottleneck features ... ...')
        print('\nCreating data generators ... ...')
        print("X_train length = ", len(self.X_train))
        print("y_train length = ", len(self.y_train))
        print("X_test length = ", len(self.X_test))
        print("y_test length = ", len(self.y_test))
        # data generators are instructions to Keras for further processing of the
        # image data (in batches) before training on the image.
        train_datagen, test_datagen = self.get_datagenerators_v1()
        # Only required if featurewise_center or featurewise_std_normalization or
        # zca_whitening are configured
        train_datagen.fit(self.X_train)
        test_datagen.fit(self.X_test)
        train_generator = train_datagen.flow(
            self.X_train,
            self.y_train,    # labels just get passed through
            batch_size=self.batch_size,
            shuffle=False,
            seed=None)
        test_generator = test_datagen.flow(
            self.X_test,
            self.y_test, # labels just get passed through
            batch_size=self.batch_size,
            shuffle=False,
            seed=None)

        print('\nLoading VGG16 model ... ...')
        model = applications.VGG16(include_top=False, weights='imagenet',input_shape=(224, 224, 3))

        # Generate bottleneck data. This is obtained by running the model on the
        # training and test data just once, recording the output (last activation
        # maps before the fully-connected layers) into separate numpy arrays.
        # This data will be used to train and validate a fully-connected model on
        # top of the stored features for computational efficiency.
        print('\nRunning train predictor and saving features ... ...')
        bottleneck_features_train = model.predict_generator(
            generator=train_generator,
            steps=self.X_train.shape[0] // self.batch_size,
            max_queue_size=self.batch_size*8,
            workers=8,
            use_multiprocessing=False,
            verbose=True)
        np.save('../../dsi-capstone-data/bottleneck_features_train.npy',
                np.array(bottleneck_features_train))
        print("Train bottleneck feature length = ", len(bottleneck_features_train))

        print('\nRunning test predictor and saving features ... ...')
        bottleneck_features_test = model.predict_generator(
            generator=test_generator,
            steps=self.X_test.shape[0] // self.batch_size,
            max_queue_size=self.batch_size*8,
            workers=8,
            use_multiprocessing=False,
            verbose=True)
        np.save('../../dsi-capstone-data/bottleneck_features_test.npy',
                np.array(bottleneck_features_test))
        print("Test bottleneck feature length = ", len(bottleneck_features_test))

        # steps_per_epoch=X_train.shape[0] // batch_size in the code above can
        # result in discarding the last few data samples if batch_size doesn't
        # divide evenly into the shape of the data array. Remove the top few samples
        # from the instance variables to keep them equal.
        self.X_train = self.X_train[:len(bottleneck_features_train)]
        self.y_train = self.y_train[:len(bottleneck_features_train)]
        self.X_test = self.X_test[:len(bottleneck_features_test)]
        self.y_test = self.y_test[:len(bottleneck_features_test)]
        pass

    def train_top_model(self):

        print('\nTraining top model ... ...')
        print('\nLoading bottleneck features ... ...')
        bottleneck_features_train = np.load('../../dsi-capstone-data/bottleneck_features_train.npy')
        bottleneck_features_test = np.load('../../dsi-capstone-data/bottleneck_features_test.npy')
        print(bottleneck_features_train.shape)
        print(bottleneck_features_test.shape)

        model = self.get_top_model(bottleneck_features_train.shape[1:])

        print('\nFitting the model ... ...')
        model.fit(x=bottleneck_features_train,
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=20,
            verbose=True,
            callbacks=None,
            validation_split=0.0,
            validation_data=(bottleneck_features_test, self.y_test),
            shuffle=False,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0)

        print('\nSaving the weights ... ...')
        model.save_weights('../../dsi-capstone-data/top_model_weights.h5')

        pass

    def fine_tune_model(self):

        print('\nFine tuning the model ... ...')
        print('\nCreating data generators ... ...')
        # data generators are instructions to Keras for further processing of the
        # image data (in batches) before training on the image.
        train_datagen, test_datagen = self.get_datagenerators_v1()
        # Only required if featurewise_center or featurewise_std_normalization or
        # zca_whitening are configured
        train_datagen.fit(self.X_train)
        test_datagen.fit(self.X_test)
        train_generator = train_datagen.flow(
            self.X_train,
            self.y_train,    # labels just get passed through
            batch_size=self.batch_size,
            shuffle=False,
            seed=None)
        test_generator = test_datagen.flow(
            self.X_test,
            self.y_test, # labels just get passed through
            batch_size=self.batch_size,
            shuffle=False,
            seed=None)

        print('\nBuild the complete VGG16 model ... ...')
        base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        top_model = self.get_top_model(input_shape=base_model.output_shape[1:])
        # note that it is necessary to start with a fully-trained model, including
        # the top model, in order to successfully do fine-tuning
        top_model.load_weights('../../dsi-capstone-data/top_model_weights.h5')

        # add the model on top of the convolutional base
        model = Model(input= base_model.input, output= top_model(base_model.output))
        print(model.summary())

        # set the first 14 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in model.layers[:15]:
            layer.trainable = False

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy'])

        # fine-tune the model
        print('\nFitting the model ... ...')
        model.fit_generator(train_generator,
            steps_per_epoch=self.X_train.shape[0] // self.batch_size,
            epochs=10,
            verbose=True,
            callbacks=None,
            validation_data=None,
            validation_steps=None,
            class_weight=None,
            max_queue_size=self.batch_size*8,
            workers=8,
            use_multiprocessing=False,
            shuffle=False,
            initial_epoch=0)

        print('\Scoring the model ... ...')
        scores = model.evaluate_generator(test_generator,
            steps=self.X_test.shape[0] // self.batch_size,
            max_queue_size=self.batch_size*8,
            workers=8,
            use_multiprocessing=False)
        print(scores)

        print('\Make predictions ... ...')
        pred = model.predict_generator(test_generator,
            steps=self.X_test.shape[0] // self.batch_size,
            max_queue_size=self.batch_size*8,
            workers=8,
            use_multiprocessing=False,
            verbose=True)
        print("Predictions: ", pred)
        print("Actual: ", self.y_test)

        pass


def main():
    # get pre-processed image and label data
    print('\nLoading numpy arrays ... ...')
    start_time = dt.datetime.now()
    X_train = np.load('../../dsi-capstone-data/processed_training_images.npy')
    X_test = np.load('../../dsi-capstone-data/processed_test_images.npy')
    y_train = np.load('../../dsi-capstone-data/training_labels.npy')
    y_test = np.load('../../dsi-capstone-data/test_labels.npy')
    stop_time = dt.datetime.now()
    print("Loading arrays took ", (stop_time - start_time).total_seconds(), "s.\n")

    model = ModelVGG16(X_train, y_train, X_test, y_test, batch_size=32)

    model.save_bottlebeck_features()

    model.train_top_model()

    model.fine_tune_model()














if __name__ == '__main__':
    main()
