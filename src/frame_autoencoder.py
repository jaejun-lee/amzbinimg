import importlib as imt
import dataset
imt.reload(dataset)

import datetime as dt
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

np.random.seed(1) # get consistent results from a stochastic training process

def plot_image_and_histogram(img):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Image')
    ax2 = fig.add_subplot(122)
    ax2.hist(img.flatten(), color='gray', bins=25)
    ax2.set_title('Histogram of pixel intensities')
    ax2.set_xlabel('Pixel intensity')
    ax2.set_ylabel('Count')
    plt.tight_layout(pad=1)

class frame_autoencoder(object):

    def __init__(self, batch_size = 128, kernel_size = 3, latent_dim = 16, layer_filters = [16, 32]):
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.layer_filters = layer_filters
    
    def load_and_condition_MNIST_data(self):
        '''
        loads and shapes MNIST image data
        input:  None
        output: X_train (2D np array), X_test (2D np array)
        '''
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        self.input_shape = (image_size, image_size, 1)
        self.image_size = image_size
        self.X_train = x_train
        self.X_test = x_test
        
    
    def load_noise_data(self):
        noise = np.random.normal(loc=0.5, scale=0.5, size=self.X_train.shape)
        X_train_noisy = self.X_train + noise
        noise = np.random.normal(loc=0.5, scale=0.5, size=self.X_test.shape)
        X_test_noisy = self.X_test + noise
        X_train_noisy = np.clip(X_train_noisy, 0., 1.)
        X_test_noisy = np.clip(X_test_noisy, 0., 1.)
        self.X_train_noisy = X_train_noisy
        self.X_test_noisy = X_test_noisy

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
                    strides=2,
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
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        x = Conv2DTranspose(filters=1,
                        kernel_size=self.kernel_size,
                        padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

        # Instantiate Decoder Model
        self.decoder = Model(latent_inputs, outputs, name='decoder')

    def load_autoencoder(self):
        # Instantiate Autoencoder Model
        self.autoencoder = Model(self.inputs, self.decoder(self.encoder(self.inputs)), name='autoencoder')
        self.autoencoder.compile(loss='mse', optimizer='adam')
    

if __name__ == '__main__':

    frame = frame_autoencoder()
    frame.load_and_condition_MNIST_data()
    frame.load_noise_data()
    frame.load_encoder()
    frame.load_decoder()
    frame.load_autoencoder()

    # Added for Tensorboard
    tensorboard = TensorBoard(log_dir='./logs_autoencoder', histogram_freq=2)
    frame.autoencoder.fit(frame.X_train_noisy,
                    frame.X_train,
                    validation_data=(frame.X_test_noisy, frame.X_test),
                    epochs=10,
                    batch_size=frame.batch_size)
    

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




