import importlib as imt
import datasets_v3
imt.reload(datasets_v3)

import datetime as dt
import tensorflow as tf

import skimage
from skimage.io import imread
from skimage.io import imsave
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Input, InputLayer
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

np.random.seed(1) # get consistent results from a stochastic training process

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


class frame_autoencoder(object):

    def __init__(self, batch_size = 64, kernel_size = 3, latent_dim = 32, layer_filters = [16, 32]):
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.layer_filters = layer_filters
    
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
    
    def load_and_condition_dataset_feature_extraction(self):
        '''
        load and shape 128x128x3 images from x_images
        '''
        x_path = "../data/x_images/"
        y_path = "../data/x_images/"
        X_train, X_test, y_train, y_test = datasets_v3.load_data(x_path, y_path)
        X_train = X_train.astype('float32') / 255 #-5
        X_test = X_test.astype('float32') / 255
        y_train = y_train.astype('float32') / 255
        y_test = y_test.astype('float32') / 255

        self.image_size = X_train.shape[1]
        self.input_shape = (self.image_size, self.image_size, 3)
        self.Xf_train = X_train
        self.Xf_test = X_test
        self.yf_train = y_train
        self.yf_test = y_test


    def build_baseline_autoencoder(self):
        # The encoder
        encoder = Sequential()
        encoder.add(InputLayer(self.input_shape))
        encoder.add(Flatten())
        encoder.add(Dense(self.latent_dim))

        # The decoder
        decoder = Sequential()
        decoder.add(InputLayer((self.latent_dim)))
        decoder.add(Dense(np.prod(self.input_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
        decoder.add(Reshape(self.input_shape))

        self.encoder = encoder
        self.decoder = decoder
    
    def load_encoder(self):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = inputs
        # Stack of Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use MaxPooling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in self.layer_filters:
            x = Conv2D(filters=filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    activation='relu',
                    padding='same')(x)
        
        # Shape info needed to build Decoder Model
        self.shape = K.int_shape(x)
        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(self.latent_dim, name='latent_vector')(x)

        # Instantiate Encoder Model
        self.encoder = Model(inputs, latent, name='encoder')
        self.inputs = inputs
    
    def load_decoder(self):
    
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        x = Dense(self.shape[1] * self.shape[2] * self.shape[3])(latent_inputs)
        x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)

        # Stack of Transposed Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use UpSampling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in self.layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=self.kernel_size,
                                strides=1,
                                activation='relu',
                                padding='same')(x)

        x = Conv2DTranspose(filters=3,
                        kernel_size=self.kernel_size,
                        padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

        # Instantiate Decoder Model
        self.decoder = Model(latent_inputs, outputs, name='decoder')

    def load_autoencoder(self):
        # Instantiate Autoencoder Model
        self.autoencoder = Model(self.inputs, self.decoder(self.encoder(self.inputs)), name='autoencoder')
        self.autoencoder.compile(loss='mse', optimizer='adam')

def plot_history(history):    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

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

def run_baseline_autoencoder():
    
    frame = frame_autoencoder(latent_dim=64)
    frame.load_and_condition_dataset_v2()
    frame.build_baseline_autoencoder()
    
    inp = Input(frame.input_shape)
    code = frame.encoder(inp)
    reconstruction = frame.decoder(code)

    autoencoder = Model(inp,reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')

    log_dir="../logs/autoencoder/train/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = autoencoder.fit(x=frame.X_train, 
                            y=frame.y_train, 
                            epochs=20, 
                            validation_split=0.2,
                            callbacks = [tensorboard])

    #history = autoencoder.fit(x=frame.X_train, y=frame.y_train, epochs=5, validation_split=0.2)
    
    log_dir="../logs/autoencoder/evaluate/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)    
    autoencoder.evaluate(x=frame.X_test, 
                            y=frame.y_test, 
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
    #frame.load_and_condition_dataset_v3()
    frame.load_encoder()
    frame.load_decoder()
    frame.load_autoencoder()

    # Added for Tensorboard
    log_dir="../logs/autoencoder/train/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = frame.autoencoder.fit(x=frame.X_train,
                    y=frame.y_train,
                    epochs=20,
                    validation_split=0.2,
                    batch_size=frame.batch_size,
                    callbacks = [tensorboard],
                    use_multiprocessing=False
    )
    
    log_dir="../logs/autoencoder/evaluate/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)    
    frame.autoencoder.evaluate(x=frame.X_test, 
                            y=frame.y_test, 
                            batch_size=frame.batch_size, 
                            verbose=1, 
                            callbacks=[tensorboard], 
                            #max_queue_size=10, 
                            #workers=1, 
                            use_multiprocessing=False
    )

    return history

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




 