import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Input, InputLayer
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers

np.random.seed(43) # get consistent results from a stochastic training process

def build_encoder_cnn_16_32_32_norm_pool(frame):
    inputs = Input(shape=frame.input_shape, name='encoder_input')
    x = inputs

    x = Conv2D(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    x = Conv2D(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    x = Conv2D(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    
    # Shape info needed to build Decoder Model
    frame.shape = K.int_shape(x)
    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(frame.latent_dim, name='latent_vector')(x)

    # Instantiate Encoder Model
    frame.encoder = Model(inputs, latent, name='encoder')
    frame.inputs = inputs
    
def build_decoder_cnn_16_32_32_norm_pool(frame):

    latent_inputs = Input(shape=(frame.latent_dim,), name='decoder_input')
    x = Dense(frame.shape[1] * frame.shape[2] * frame.shape[3])(latent_inputs)
    x = Reshape((frame.shape[1], frame.shape[2], frame.shape[3]))(x)


    x = Conv2DTranspose(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=3, kernel_size=frame.kernel_size, padding='same')(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)

    # Instantiate Decoder Model
    frame.decoder = Model(latent_inputs, outputs, name='decoder')

def build_encoder_baseline(frame):

    encoder = Sequential()
    encoder.add(InputLayer(frame.input_shape))
    encoder.add(Flatten())
    encoder.add(Dense(frame.latent_dim))
    frame.encoder = encoder

def build_decoder_baseline(frame):

    decoder = Sequential()
    decoder.add(InputLayer((frame.latent_dim)))
    decoder.add(Dense(np.prod(frame.input_shape)))
    decoder.add(Reshape(frame.input_shape))
    frame.decoder = decoder



def build_autoencoder(frame):
    frame.model_autoencoder = Model(frame.inputs, frame.decoder(frame.encoder(frame.inputs)), name='autoencoder')
    optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False,
                    name='Adam'
                )
    frame.model_autoencoder.compile(loss='mse', optimizer=optimizer)

def build_encoder_cnn_baseline(frame):
        inputs = Input(shape=frame.input_shape, name='encoder_input')
        x = inputs
        # Stack of Conv2D blocks
        # Notes:
        # 1) Use Batch Normalization before ReLU on deep networks
        # 2) Use MaxPooling2D as alternative to strides>1
        # - faster but not as good as strides>1
        for filters in frame.layer_filters:
            x = Conv2D(filters=filters,
                    kernel_size=frame.kernel_size,
                    strides=1,
                    activation='relu',
                    padding='same')(x)
        
        # Shape info needed to build Decoder Model
        frame.shape = K.int_shape(x)
        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(frame.latent_dim, name='latent_vector')(x)

        # Instantiate Encoder Model
        frame.encoder = Model(inputs, latent, name='encoder')
        frame.inputs = inputs

def build_decoder_cnn_baseline(frame):
    
        latent_inputs = Input(shape=(frame.latent_dim,), name='decoder_input')
        x = Dense(frame.shape[1] * frame.shape[2] * frame.shape[3])(latent_inputs)
        x = Reshape((frame.shape[1], frame.shape[2], frame.shape[3]))(x)

        for filters in frame.layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=self.kernel_size,
                                strides=1,
                                activation='relu',
                                padding='same')(x)

        x = Conv2DTranspose(filters=3,
                        kernel_size=frame.kernel_size,
                        padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)

        # Instantiate Decoder Model
        frame.decoder = Model(latent_inputs, outputs, name='decoder')

def build_decoder_predict(frame):

    latent_inputs = Input(shape=(frame.latent_dim,), name='decoder_input')
    x = Dense(frame.latent_dim, activation='relu')(latent_inputs)
    x = Dropout(0.2)(x)
    x = Dense(frame.latent_dim, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(6, activation='softmax')(x)
    frame.decoder = Model(latent_inputs, outputs, name='decoder_predicter')

def build_model_predict(frame):

    frame.model_predict = Model(frame.inputs, frame.decoder(frame.encoder(frame.inputs)), name='model_predict')
    optimizer = optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'
    )
    frame.model_predict.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy', soft_rmse])


def build_encoder_cnn_16_32_64_512(frame):
    #score 0.24 with good loss history

    inputs = Input(shape=frame.input_shape, name='encoder_input')
    x = inputs

    x = Conv2D(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    #x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    x = Conv2D(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    #x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    x = Conv2D(filters=64, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    
    # Shape info needed to build Decoder Model
    frame.shape = K.int_shape(x)
    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(frame.latent_dim, name='latent_vector')(x)

    # Instantiate Encoder Model
    frame.encoder = Model(inputs, latent, name='encoder')
    frame.inputs = inputs

def build_encoder_cnn_16_32_64_512(frame):

    latent_inputs = Input(shape=(frame.latent_dim,), name='decoder_input')
    x = Dense(frame.shape[1] * frame.shape[2] * frame.shape[3])(latent_inputs)
    x = Reshape((frame.shape[1], frame.shape[2], frame.shape[3]))(x)


    x = Conv2DTranspose(filters=64, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    #x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    #x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=3, kernel_size=frame.kernel_size, padding='same')(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)

    # Instantiate Decoder Model
    frame.decoder = Model(latent_inputs, outputs, name='decoder')

def build_autoencoder_binary_crossentropy(frame):
    frame.autoencoder = Model(frame.inputs, frame.decoder(frame.encoder(frame.inputs)), name='autoencoder')
    optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False,
                    name='Adam'
                )
    frame.autoencoder.compile(loss='binary_crossentropy', optimizer=optimizer)


def soft_rmse(y_true, y_pred):
    return K.sqrt(  K.mean(K.cast_to_floatx ( K.square( K.argmax(y_true) - K.argmax(y_pred) )), axis=-1) )


def build_encoder_cnn_16_32_16(frame):
    inputs = Input(shape=frame.input_shape, name='encoder_input')
    x = inputs

    x = Conv2D(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    x = Conv2D(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    x = Conv2D(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    
    # Shape info needed to build Decoder Model
    frame.shape = K.int_shape(x)
    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(frame.latent_dim, name='latent_vector')(x)

    # Instantiate Encoder Model
    frame.encoder = Model(inputs, latent, name='encoder')
    frame.inputs = inputs

def build_decoder_cnn_16_32_16(frame):

    latent_inputs = Input(shape=(frame.latent_dim,), name='decoder_input')
    x = Dense(frame.shape[1] * frame.shape[2] * frame.shape[3])(latent_inputs)
    x = Reshape((frame.shape[1], frame.shape[2], frame.shape[3]))(x)


    x = Conv2DTranspose(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=32, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=16, kernel_size=frame.kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization(trainable=False)(x)
    x = relu(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(filters=3, kernel_size=frame.kernel_size, padding='same')(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)

    # Instantiate Decoder Model
    frame.decoder = Model(latent_inputs, outputs, name='decoder')





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




 