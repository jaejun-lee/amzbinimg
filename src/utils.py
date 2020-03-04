import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import numpy as np


### Custome Metrics
def soft_rmse(y_true, y_pred):
    '''caculate RMSE for categorical predictions. 
    '''
    return K.sqrt(  K.mean(K.cast_to_floatx ( K.square( K.argmax(y_true) - K.argmax(y_pred) )), axis=-1) )

def soft_acc(y_true, y_pred):
    '''caculate accuracy for RMSE loss function prediction.
    '''
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

### Plot
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

### configure training layer
def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True
    
def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

### Image Processing
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