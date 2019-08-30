
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, Model, model_from_json 
from keras.layers import (
    Input,
    Activation,
    concatenate,
    merge,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling2D,
    Flatten,
    LeakyReLU
    )
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import (
    Convolution2D)
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

weight_decay = 1e-5
K.set_image_dim_ordering('tf')

def _conv_bn_relu(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', 
                               border_mode='same', bias = False)(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        return act_a
    return f
    
def _conv_bn_relu_x2(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',bias = False,
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        conv_b = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',bias = False,
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(act_a)
        norm_b = BatchNormalization()(conv_b)
        act_b = Activation(activation = 'relu')(norm_b)
        return act_b
    return f

def U_net_base(input, nb_filter = 64):
    block1 = _conv_bn_relu_x2(nb_filter,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(nb_filter,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(nb_filter,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu_x2(nb_filter,3,3)(pool3)
    x = UpSampling2D(size=(2, 2))
    up4 = concatenate([x(block4), block3], axis=-1)
    # =========================================================================
    block5 = _conv_bn_relu_x2(nb_filter,3,3)(up4)
    up5 = concatenate([UpSampling2D(size=(2, 2))(block5), block2], axis=-1)
    # =========================================================================
    block6 = _conv_bn_relu_x2(nb_filter,3,3)(up5)
    up6 = concatenate([UpSampling2D(size=(2, 2))(block6), block1], axis=-1)
    # =========================================================================
    block7 = _conv_bn_relu(nb_filter,3,3)(up6)
    return block7

def Unet (input_dim,weights):
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = U_net_base (input_, nb_filter = 64 )
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    Unet = Model (input = input_, output = density_pred)
    opt = RMSprop(1e-3)
    Unet.compile(optimizer = opt, loss = 'mse')
    Unet.load_weights(weights)#weights from Xie et al.
    return Unet

def DRDCNN(input_dim,weights):
    unet_model = Unet(input_dim,weights)
    x = unet_model.output
    x = BatchNormalization(name="batch_normalization_14")(x)
    x = Convolution2D(64,[3,3], kernel_regularizer=l2(0.0001), padding='same',name="block1_1")(x)
    x = LeakyReLU(alpha=0.1,name="activation_18")(x)
    x = MaxPooling2D(pool_size=(28 ,28),strides=4,name="max_pooling2d_1_1")(x)
    x = BatchNormalization(name="batch_normalization_2_1")(x)
    x = Convolution2D(32,[3,3],kernel_regularizer=l2(0.0001),  padding='same',name="block2_1")(x)
    x = LeakyReLU(alpha=0.1,name="activation_2_1")(x)
    x = MaxPooling2D(pool_size=(7, 7),strides=4,name="max_pooling2d_2_1")(x)
    x = BatchNormalization(name="batch_normalization_4_1")(x)
    x = Dense(4096,kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(alpha=0.1,name="activation_4_1")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2,name="max_pooling2d_4_1")(x)
    x = BatchNormalization(name="batch_normalization_5_1")(x)
    x = Dense(32,kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(alpha=0.1,name="activation_5_1")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2,name="max_pooling2d_5_1")(x)
    x = BatchNormalization(name="batch_normalization_6_1")(x)
    x = Flatten()(x)
    x = Dense(1,kernel_regularizer=l2(0.0001),activation="relu")(x)
    DRDCNN = Model(inputs=unet_model.input, outputs=x)
    for layer in unet_model.layers:
        layer.trainable = False
    DRDCNN.compile(loss='mean_squared_error', 
        optimizer=keras.optimizers.Adam(lr=0.1,decay=1e-3), 
        metrics=['accuracy'])
    return DRDCNN

def FM(input_dim,weights):
    input_img = Input(shape=(input_dim),name = "Input")
    x = Convolution2D(16, (3, 3), activation='relu', padding='same',name = "Encode_Layer1")(input_img)
    x = MaxPooling2D((2, 2), padding='same',name = "Encode_Layer2")(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same',name = "Encode_Layer3")(x)
    x = MaxPooling2D((2, 2), padding='same',name = "Encode_Layer4")(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same',name = "Encode_Layer5")(x)
    x = MaxPooling2D((2, 2), padding='same',name = "Encode_Layer6")(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same',name = "Decode_Layer1")(x)
    x = UpSampling2D((2, 2),name = "Decode_Layer2")(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same',name = "Decode_Layer3")(x)
    x = UpSampling2D((2, 2),name = "Decode_Layer4")(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same',name = "Decode_Layer5")(x)
    x = UpSampling2D((2, 2),name = "Decode_Layer6")(x)
    x = Convolution2D(1, (3, 3), activation='relu', padding='same',name = "Output")(x)
    FM = Model(input_img, x)
    FM.compile(optimizer='adadelta', loss='mean_squared_error')
    FM.load_weights(weights)
    return FM

def FM_pre(model,weights):
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FM = model_from_json(loaded_model_json)
    FM.load_weights(weights)
    return FM

def FRDCNN(model,weights):
    FM_model = FM_pre(model,weights)
    x = FM_model.output
    x = BatchNormalization(name="batch_normalization_14")(x)
    x = Convolution2D(64,[3,3], kernel_regularizer=l2(0.0001), padding='same',name="block1_1")(x)
    x = LeakyReLU(alpha=0.1,name="activation_18")(x)
    x = MaxPooling2D(pool_size=(14 ,14),strides=4,name="max_pooling2d_1_1")(x)
    x = BatchNormalization(name="batch_normalization_2_1")(x)
    x = Convolution2D(32,[3,3],kernel_regularizer=l2(0.0001),  padding='same',name="block2_1")(x)
    x = LeakyReLU(alpha=0.1,name="activation_2_1")(x)
    x = MaxPooling2D(pool_size=(7, 7),strides=4,name="max_pooling2d_2_1")(x)
    x = BatchNormalization(name="batch_normalization_3_1")(x)
    x = Convolution2D(16,[3,3],kernel_regularizer=l2(0.0001),  padding='same',name="block3_1")(x)
    x = LeakyReLU(alpha=0.1,name="activation_3_1")(x)
    x = MaxPooling2D(pool_size=(7, 7),strides=4,name="max_pooling2d_3_1")(x)
    x = BatchNormalization(name="batch_normalization_4_1")(x)
    x = Dense(4096,kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(alpha=0.1,name="activation_4_1")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2,name="max_pooling2d_4_1")(x)
    x = BatchNormalization(name="batch_normalization_5_1")(x)
    x = Dense(1024,kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(alpha=0.1,name="activation_5_1")(x)
    x = BatchNormalization(name="batch_normalization_6_1")(x)
    x = Flatten()(x)
    x = Dense(1,kernel_regularizer=l2(0.0001),activation="relu")(x)
    FRDCNN = Model(inputs=FM_model.input, outputs=x)
    for layer in FM_model.layers:
        layer.trainable = False
    FRDCNN.compile(loss='mean_squared_error', 
        optimizer=keras.optimizers.Adam(lr=0.1,decay=1e-3), 
        metrics=['accuracy'])
    return FRDCNN

def CRDCNN(input_dim,model,weights_unet,weights_FM):
    unet_model = Unet(input_dim,weights_unet)
    FM_model = FM_pre(model,weights_FM)
    input1 = unet_model.output
    input2 = FM_model.output
    x1 = BatchNormalization(name="input1_batch_normalization_14")(input1)
    x1 = Convolution2D(64,[3,3],  padding='same',kernel_regularizer=l2(0.001),name="input1_conv2d_14")(x1)
    x2 = BatchNormalization(name="input2_batch_normalization_14")(input2)
    x2 = Convolution2D(64,[3,3],  padding='same',kernel_regularizer=l2(0.001),name="input2_conv2d_14")(x2)
    x = concatenate([x1,x2],name="concat_features")
    x = BatchNormalization(name="batch_normalization_14")(x)
    x = Convolution2D(64,[3,3], kernel_regularizer=l2(0.0001), padding='same',name="block1_1")(x)
    x = LeakyReLU(alpha=0.1,name="activation_18")(x)
    x = MaxPooling2D(pool_size=(28 ,28),strides=4,name="max_pooling2d_1_1")(x)
    x = BatchNormalization(name="batch_normalization_2_1")(x)
    x = Convolution2D(32,[3,3],kernel_regularizer=l2(0.0001),  padding='same',name="block2_1")(x)
    x = LeakyReLU(alpha=0.1,name="activation_2_1")(x)
    x = MaxPooling2D(pool_size=(7, 7),strides=4,name="max_pooling2d_2_1")(x)
    x = BatchNormalization(name="batch_normalization_4_1")(x)
    x = Dense(4096,kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(alpha=0.1,name="activation_4_1")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2,name="max_pooling2d_4_1")(x)
    x = BatchNormalization(name="batch_normalization_5_1")(x)
    x = Dense(32,kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(alpha=0.1,name="activation_5_1")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=2,name="max_pooling2d_5_1")(x)
    x = BatchNormalization(name="batch_normalization_6_1")(x)
    x = Flatten()(x)
    x = Dense(1,kernel_regularizer=l2(0.0001))(x)
    x = LeakyReLU(alpha=0.1,name="activation_7_1")(x)
    CRDCNN = Model(inputs=[unet_model.input,FM_model.input], outputs=x)
    for layer in FM_model.layers:
        layer.trainable = False
    for layer in unet_model.layers:
        layer.trainable = False
    CRDCNN.compile(loss='mean_squared_error', 
        optimizer=keras.optimizers.Adam(lr=0.01,decay=1e-3), 
        metrics=['accuracy'])
    return CRDCNN
