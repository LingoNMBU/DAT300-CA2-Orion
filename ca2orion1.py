import h5py
import numpy as np
import random

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
    
    outputs = Conv2D(n_classes, (1, 1), activation=class_activation)(c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model


model = Model_unet(model_function=get_unet_vgg16, n_filters=32, dropout = 0.4, batch_size=40)
model.fit()