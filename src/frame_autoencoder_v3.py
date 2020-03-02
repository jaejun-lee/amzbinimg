import importlib as imt
import datasets_v3
<<<<<<< HEAD
import datasets_v0
import catalog
imt.reload(datasets_v3)
imt.reload(datasets_v0)
=======
import catalog
imt.reload(datasets_v3)
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe
imt.reload(catalog)

import datetime as dt

import skimage
from skimage.io import imread
from skimage.io import imsave
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Activation, Dense, Input, InputLayer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D

from tensorflow.keras.activations import relu

from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.callbacks import TensorBoard



np.random.seed(43) # get consistent results from a stochastic training process

class frame_autoencoder(object):

    def __init__(self, batch_size = 128, kernel_size = 3, latent_dim = 128, layer_filters = [16, 32]):
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.layer_filters = layer_filters
        self.shape = None
        self.model_autoencoder = None
        self.model_predict = None
        self.encoder = None
        self.decoder = None
    
    def load_and_condition_dataset_denoise(self, x_path, y_path):
        '''
        load and shape 128x128x3 images from x_images
        '''
        X_train, X_test, y_train, y_test = datasets_v3.load_data(x_path, y_path)
        X_train = X_train.astype('float32') / 255 #-5
        X_test = X_test.astype('float32') / 255
        y_train = y_train.astype('float32') / 255
        y_test = y_test.astype('float32') / 255

        self.image_size = X_train.shape[1]
        self.input_shape = (self.image_size, self.image_size, 3)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def load_and_condition_dataset_reco(self):
        '''
        load and shape 128x128x3 images from x_images
        '''
        x_path = "../data/x_images/"
        y_path = "../data/x_images/"
        X_train, X_test, y_train, y_test = datasets_v3.load_data(x_path, y_path)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        y_train = y_train.astype('float32') / 255
        y_test = y_test.astype('float32') / 255

        self.image_size = X_train.shape[1]
        self.input_shape = (self.image_size, self.image_size, 3)
        self.Xf_train = X_train
        self.Xf_test = X_test
        self.yf_train = y_train
        self.yf_test = y_test
    
    def load_encoder(self, builder):
        builder(self)
    
    def load_decoder(self, builder):
        builder(self)

    def load_model_autoencoder(self, builder):
        builder(self)

    def load_model_predict(self, builder):
        builder(self)

<<<<<<< HEAD

    def load_datagenerators(self, X_train, y_train, X_test, y_test, input_size = (128, 128)):
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

=======
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe
def run_baseline_autoencoder():
    
    frame = frame_autoencoder(latent_dim=64)
    frame.load_and_condition_dataset_reco()
    frame.load_encoder(catalog.build_encoder_baseline)
    frame.load_decoder(catalog.build_decoder_baseline)
    
    inp = Input(frame.input_shape)
    code = frame.encoder(inp)
    reconstruction = frame.decoder(code)

<<<<<<< HEAD
    frame.model_autoencoder = Model(inp,reconstruction)
    frame.model_autoencoder.compile(optimizer='adamax', loss='mse')

    log_dir="../logs/autoencoder/train/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = frame.model_autoencoder.fit(x=frame.Xf_train, 
=======
    frame.autoencoder = Model(inp,reconstruction)
    frame.autoencoder.compile(optimizer='adamax', loss='mse')

    log_dir="../logs/autoencoder/train/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = frame.autoencoder.fit(x=frame.Xf_train, 
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe
                            y=frame.yf_train, 
                            epochs=20, 
                            validation_split=0.2,
                            callbacks = [tensorboard])

    #history = autoencoder.fit(x=frame.X_train, y=frame.y_train, epochs=5, validation_split=0.2)
    
    log_dir="../logs/autoencoder/evaluate/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)    
<<<<<<< HEAD
    frame.model_autoencoder.evaluate(x=frame.Xf_test, 
=======
    frame.autoencoder.evaluate(x=frame.Xf_test, 
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe
                            y=frame.yf_test, 
                            #batch_size=frame.batch_size, 
                            verbose=1, 
                            callbacks=[tensorboard], 
                            #max_queue_size=10, 
                            #workers=1, 
                            use_multiprocessing=False
    )

    return history

def run_convnet_autoencoder():

    frame = frame_autoencoder(batch_size = 64, 
                            kernel_size = 3, 
                            latent_dim = 32, 
                            layer_filters = [16, 32]
    )
    frame.load_and_condition_dataset_reco()
<<<<<<< HEAD
    frame.load_encoder(catalog.build_encoder_cnn_16_32_32_norm_pool)
    frame.load_decoder(catalog.build_decoder_cnn_16_32_32_norm_pool)
    frame.load_model_autoencoder(catalog.build_autoencoder)
=======
    frame.load_encoder(catalog.build_encoder_16_32_32_batch_pool)
    frame.load_decoder(catalog.build_decoder_16_32_32_batch_pool)
    frame.load_autoencoder(catalog.build_autoencoder)
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe

    # Added for Tensorboard
    log_dir="../logs/autoencoder/train/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
<<<<<<< HEAD
    history = frame.model_autoencoder.fit(x=frame.Xf_train,
=======
    history = frame.autoencoder.fit(x=frame.Xf_train,
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe
                    y=frame.yf_train,
                    epochs=20,
                    validation_split=0.2,
                    batch_size=frame.batch_size,
                    callbacks = [tensorboard],
                    use_multiprocessing=False
    )
    
    log_dir="../logs/autoencoder/evaluate/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)    
<<<<<<< HEAD
    frame.model_autoencoder.evaluate(x=frame.Xf_test, 
=======
    frame.autoencoder.evaluate(x=frame.Xf_test, 
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe
                            y=frame.yf_test, 
                            batch_size=frame.batch_size, 
                            verbose=1, 
                            callbacks=[tensorboard], 
                            #max_queue_size=10, 
                            #workers=1, 
                            use_multiprocessing=False
    )

    return history

<<<<<<< HEAD

def run_prediction_model():
    # load dataset
    X_train, X_test, y_train, y_test = datasets_v0.load_data()

    frame = frame_autoencoder(latent_dim=256)

    # Prepare Datasets
    frame.load_datagenerators(X_train, y_train, X_test, y_test, input_size = (128, 128))
    frame.input_shape = (128, 128, 3)

    # load models
    frame.load_encoder(catalog.build_encoder_cnn_16_32_32_norm_pool)
    frame.load_decoder(catalog.build_decoder_predict)
    frame.load_model_predict(catalog.build_model_predict)

    # set trainable layer to top layers only
    #change_trainable_layers(model, 132)

    print('\nFitting the model ... ...')
    log_dir="../logs/fit/xception/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = frame.model_predict.fit(
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
    frame.model_predict.evaluate(
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

    pred = frame.model_predict.predict(x=frame.test_generator,
            steps=frame.STEP_SIZE_TEST,
            max_queue_size=frame.batch_size*8,
            #workers=8,
            use_multiprocessing=False,
            verbose=True)

    return history, pred

=======
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe
def plot_history(history):    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_image_and_histogram(img):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax1.set_title('Image')
    ax2 = fig.add_subplot(122)
    ax2.hist(img[:,:,0].flatten(), color='red', bins=25)
    ax2.hist(img[:,:,1].flatten(), color='green', bins=25)
    ax2.hist(img[:,:,2].flatten(), color='blue', bins=25)
    ax2.set_title('Histogram of pixel intensities')
    ax2.set_xlabel('Pixel intensity')
    ax2.set_ylabel('Count')
    plt.tight_layout(pad=1)

def visualize(img, encoder, decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(reco)
    plt.show()


def visualize_features(img, encoder, decoder):
    code = encoder.predict(img[None])[0]
    
    images_per_row = 16
    num_of_features = code.shape[0]
    n_cols =  num_of_features // images_per_row
    display_grid = np.zeros(images_per_row, n_cols)
    
    for row in range(images_per_row):
        for col in range(n_cols):
            mask = np.zeros(num_of_features)
            mask[row * n_cols + cos] = 1
            new_code = code.copy()
            new_code = new_code * mask
            reco = decoder.predict(new_code[None])[0]
            display_grid[row, col] = reco
    
    plt.figure(figsize= (images_per_row, n_cols))
    plt.title("Feature Representation by Decoder")
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
<<<<<<< HEAD

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True
    
def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")
=======
>>>>>>> 5f7a43753f7bf414ef6814a0cdbe00a0cae55bbe


if __name__ == '__main__':


    

    pass




























    # for i in range(5):
    #     img = frame.X_test[i]
    #     visualize(img,frame.encoder,frame.decoder)
    
    #x_decoded = autoencoder.predict(frame.x_test_noisy)

    # rows, cols = 10, 30
    # num = rows * cols
    # imgs = np.concatenate([frame.x_test[:num], frame.x_test_noisy[:num], x_decoded[:num]])
    # imgs = imgs.reshape((rows * 3, cols, frame.image_size, frame.image_size))
    # imgs = np.vstack(np.split(imgs, rows, axis=1))
    # imgs = imgs.reshape((rows * 3, -1, frame.image_size, frame.image_size))
    # imgs = np.vstack([np.hstack(i) for i in imgs])
    # imgs = (imgs * 255).astype(np.uint8)

    # plt.figure(figsize=(12,12))
    # plt.axis('off')
    # plt.title('Original images: top rows, '
    #         'Corrupted Input: middle rows, '
    #         'Denoised Input:  third rows', fontsize=14)
    # plt.imshow(imgs, interpolation='none', cmap='gray')
    # plt.show()




 