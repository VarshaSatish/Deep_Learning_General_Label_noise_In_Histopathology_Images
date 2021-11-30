from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist, cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from loader import *
from classification_models.keras import Classifiers
from resnet_model import ResNet18

CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "./data/"

def get_data(dataset='BACH', noiserate=0.1, label_noise_rate=0):

   
    class_names =['Benign','Invasive','Normal']
    num_train_samples = 225
    num_val_samples = 75
    # label_noise_rate = 0.1
    (X_train0, y_train0), (X_test, y_test) = load_data(3,'./BACH_dataset/train','./BACH_dataset/validation',class_names,num_train_samples,num_val_samples,label_noise_rate)
    
    class_names1 =['InSitu']
    num_train_samples1 = 100
    num_val_samples1 = 0
    (X_train1, y_train1), (X_test1, y_test1) = load_data(1,'./BACH_dataset/noise',None,class_names1,num_train_samples1,num_val_samples1)

    # if dataset == 'cifar10-cifar100':
    #     (X_train0, y_train0), (X_test, y_test) = cifar10.load_data()
    #     (X_train1, y_train1), (X_test1, y_test1) = cifar100.load_data()

    X_train_openset = X_train1[np.random.choice(X_train1.shape[0], int(X_train0.shape[0]*noiserate), replace=False), :]
    # print(X_train_openset.shape)
    # y_train_openset = np.repeat(range(3), int(X_train0.shape[0]*noiserate/3)).reshape(-1,1)
    # y_train_openset = np.repeat(np.random.choice(0,1,2), X_train_openset.shape[0]/3).reshape(-1,1)
    y_train_openset = np.random.choice(3,X_train_openset.shape[0]).reshape(-1,1)
    # print(y_train_openset.shape)
    # print('X;',X_train_openset.shape)
    # print('Y;',y_train_openset.shape)
    # count_lnoise = int(label_noise_rate*X_train0.shape[0])

    X_train_with_lnoise = np.roll(X_train0[0:int(X_train0.shape[0]*(1-noiserate))], -int(num_train_samples*label_noise_rate))
    y_train_with_lnoise = np.roll(y_train0[0:int(X_train0.shape[0]*(1-noiserate))], -int(num_train_samples*label_noise_rate))
    # print('l-X:',X_train_with_lnoise.shape)
    # print('lnoise',y_train_with_lnoise.shape)
   
    # X_train = np.concatenate((X_train0[0:int(X_train0.shape[0]*(1-noiserate))], X_train_openset), axis=0)
    # y_train = np.concatenate((y_train0[0:int(X_train0.shape[0]*(1-noiserate))], y_train_openset), axis=0)
    
    X_train = np.concatenate((X_train_with_lnoise, X_train_openset), axis=0)
    y_train = np.concatenate((y_train_with_lnoise, y_train_openset), axis=0)
    
    # print(X_train.shape)
    # print(y_train.shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = (X_train/255.0) - (1.0 - CLIP_MAX)
    X_test = (X_test/255.0) - (1.0 - CLIP_MAX)

    np.save('./data/X_train.npy', X_train)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/X_test.npy', X_test)
    np.save('./data/y_test.npy', y_test)

    return X_train, y_train, X_test, y_test

def get_model(dataset='cifar10'):
    #if dataset == 'cifar10':
    layers = [
        Conv2D(64, (3, 3), padding='same', input_shape=(150, 150, 3)),  # 0
        BatchNormalization(),
        Activation('relu'),  # 1
        Conv2D(64, (3, 3), padding='same'),  # 2
        BatchNormalization(),
        Activation('relu'),  # 3
        MaxPooling2D(pool_size=(2, 2)),  # 4
        Conv2D(128, (3, 3), padding='same'),  # 5
        BatchNormalization(),
        Activation('relu'),  # 6
        Conv2D(128, (3, 3), padding='same'),  # 7
        BatchNormalization(),
        Activation('relu'),  # 8
        MaxPooling2D(pool_size=(2, 2)),  # 9
        Conv2D(196, (3, 3), padding='same'),  # 10
        BatchNormalization(),
        Activation('relu'),  # 11
        Conv2D(196, (3, 3), padding='same'),  # 12
        BatchNormalization(),
        Activation('relu'),  # 13
        MaxPooling2D(pool_size=(2, 2)),  # 14
        Flatten(),  # 15
        Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model

def get_resnet():
    # n_classes = 256
    # base_model = ResNet18(inputs=(224,224,3), num_classes=3)

    ResNet18, preprocess_input = Classifiers.get('resnet18')

    n_classes = 3

    # build model
    base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])

    return model
