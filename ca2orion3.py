import h5py
import numpy as np
import random
import tensorflow 
import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from keras import backend as K

with h5py.File('../ca2data/train.h5', 'r') as hf:
    X_data = hf['X'][:]
    y_data = hf['y'][:]
with h5py.File('../ca2data/test.h5', 'r') as hf:
    kaggle = hf["X"][:]
    
    
y_data[y_data != 0] = 1 
    

np.random.seed(123)

indices = np.random.permutation(X_data.shape[0])

training_idx, test_idx = indices[:int(0.80*len(X_data))], indices[int(0.80*len(X_data)):]

X_train, X_test, y_train, y_test = X_data[training_idx,:], X_data[test_idx,:], y_data[training_idx,:], y_data[test_idx,:]



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def f1_loss(true, pred):
    true= K.cast(true, tensorflow.float32)
    ground_positives = K.sum(true, axis=0) + K.epsilon()       # = TP + FN
    pred_positives = K.sum(pred, axis=0) + K.epsilon()         # = TP + FP
    true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    #soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    
    #f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    
    return 1 - K.mean(f1)



"""
Version of U-Net with dropout and size preservation (padding= 'same')
""" 

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x



def get_unet_vgg16(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 2, class_activation= 'sigmoid'):
    
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_img, classes = 2)

    # Contracting Path
    c1 = vgg16.get_layer("block1_conv2").output         
    c2 = vgg16.get_layer("block2_conv2").output         
    c3 = vgg16.get_layer("block3_conv3").output         
    c4 = vgg16.get_layer("block4_conv3").output         
    
    # Bridge
    c5 = vgg16.get_layer("block5_conv3").output         
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation=class_activation)(c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model


import keras_tuner
from tensorflow import keras
from numpy.random import seed
def build_tuning_model(hp):
    
    model = get_unet_vgg16(input_img = Input(shape=(128,128,3)),                          
                           n_filters= hp.Int('n_filters', min_value=15, max_value=80, sampling="linear"), 
                           dropout = hp.Float('dropout', min_value=0.0, max_value=0.35, sampling="linear"),
                           batchnorm = True, n_classes = 2, class_activation= 'sigmoid')
    
    model.compile(optimizer = Adam(learning_rate = hp.Float("lr", min_value=1e-4, max_value=0.5e-1, sampling="log") ),
                  loss = "binary_crossentropy", 
                  metrics = ['accuracy', f1_m])
    
    return model


tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_tuning_model,
    objective="val_loss",
    max_trials=100,
    executions_per_trial=1,
    overwrite=False,
    directory="",
    project_name="exp_unet_bay",
    seed=102
)
tuner.search_space_summary()
tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size = 50,
             callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])


