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


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, strides = (1,1)):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same', strides=strides)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same', strides=strides)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def get_unet2(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 2, class_activation= 'sigmoid'):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    #p1 = MaxPooling2D((2, 2))(c1)
    p1 = conv2d_block(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, strides=(1,1))
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = conv2d_block((c2), n_filters * 2, kernel_size = 3, batchnorm = batchnorm, strides=(1,1))
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = conv2d_block((c3), n_filters * 4, kernel_size = 3, batchnorm = batchnorm, strides=(2,2))
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = conv2d_block((c4), n_filters * 8, kernel_size = 3, batchnorm = batchnorm, strides=(2,2))
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
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
    
    outputs = Conv2D(n_classes, (1, 1), activation=class_activation)(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras_tuner
from tensorflow import keras
from numpy.random import seed
seed(102)

from sklearn import model_selection
import keras_tuner

def build_tuning_model(hp):
    
    model = get_unet2(input_img = Input(shape=(128,128,3)),                          
                           n_filters= hp.Int('n_filters', min_value=15, max_value=80, sampling="linear"), 
                           dropout = hp.Float('dropout', min_value=0.0, max_value=0.4, sampling="linear"),
                           batchnorm = True, n_classes = 1, class_activation= 'sigmoid')
    
    model.compile(optimizer = Adam(learning_rate = hp.Float("lr", min_value=1e-5, max_value=0.5e-1, sampling="log")),
                  loss = "binary_crossentropy", 
                  metrics = ['accuracy', f1_m])
    
    return model

class CVTuner(keras_tuner.engine.tuner.Tuner):
  def run_trial(self, trial, x, y, batch_size=40, epochs=1):
    cv = model_selection.KFold(5)
    val_losses = []
    for train_indices, test_indices in cv.split(x):
      x_train, x_test = x[train_indices], x[test_indices]
      y_train, y_test = y[train_indices], y[test_indices]
      model = self.hypermodel.build(trial.hyperparameters)
      model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
      val_losses.append(model.evaluate(x_test, y_test))
    self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
    self.save_model(trial.trial_id, model)   
    

tuner = CVTuner(hypermodel=build_tuning_model,
                oracle=keras_tuner.oracles.HyperbandOracle(objective='val_loss', max_epochs=50,))

tuner.search_space_summary()
x, y = X_train, y_train  # NumPy data
tuner.search(x, y, batch_size=64, epochs=30, overwrite=True)
