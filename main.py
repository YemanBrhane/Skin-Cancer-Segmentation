#=======================================================================
#
# This is the file for main script
#
#=======================================================================

#-----------------------------------------------------------------------
# Libs
#	Importing the required libraries
#	We are using Keras on top of tensorflow	
#-----------------------------------------------------------------------
from __future__ import print_function
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
K.set_image_dim_ordering('th')  



#-----------------------------------------------------------------------
# Variables
#-----------------------------------------------------------------------
# image size
numRows = 64*3
numCols = 80*3
# number of epochs
numEpochs = 500
# batchsize
batchSize = 10
# for computing dice coefficients
smooth = 1.



#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
# defining function to load the vectorized training data and the vectorized training mask data
def load_train_data(): 
    trainData = np.load('trainData.npy')
    trainMaskData = np.load('trainMaskData.npy')
    return trainData, trainMaskData


# defining function to load the vectorized testing data	
def load_test_data(): 
    testData = np.load('testData.npy')
    return testData


# defining function to calculate the dice Coefficient
def compute_dice_coef(yTrue, yPred): 
    yLabelFlatten = K.flatten(yTrue)
    yPredictFlatten = K.flatten(yPred)
    intersection = K.sum(yLabelFlatten * yPredictFlatten)
    return (2. * intersection + smooth) / (K.sum(yLabelFlatten) + K.sum(yPredictFlatten) + smooth)

	
# defining function to calculate the dice Coefficient Loss	
def compute_dice_coef_loss(yTrue, yPred): 
    return -compute_dice_coef(yTrue, yPred)


# defining Unet Model
def get_unet():
    inputs = Input((1, numRows, numCols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-6), loss=compute_dice_coef_loss, metrics=[compute_dice_coef]) ##Compile the net with Adam optimizer and dicecoef as the loss function
    print('Loading saved weights...')
    # this load model is used to resume the training if an existing previously trained weight profile of the network exists
    model.load_weights('unet_last.hdf5')
    return model

	
# defining function to resize the images to the defined numRows and columns
def preprocess_data(image): 
    imagePre = np.ndarray((image.shape[0], image.shape[1], numRows, numCols), dtype=np.uint8)
    for i in range(image.shape[0]):
        imagePre[i, 0] = cv2.resize(image[i, 0], (numCols, numRows), interpolation=cv2.INTER_CUBIC)
    return imagePre

	
	
#-----------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------	
if __name__ == '__main__':

	# loading and preprocess the training data and the training mask data
    print('Loading and preprocessinging train data')
    trainData, trainMaskData = load_train_data()
    trainData = preprocess_data(trainData)
    trainMaskData = preprocess_data(trainMaskData) 

	# changing the file type to float32
    trainData = trainData.astype('float32') 
	# mean for data centering
    mean = np.mean(trainData)
	# std for data normalization	
    std = np.std(trainData)  

	# init train data
    trainData -= mean
    trainData /= std
    trainMaskData = trainMaskData.astype('float32')
	# binarizing the train mask data pixel values from 0-1
    trainMaskData /= 255. 

	# create and compile model
    print('Creating and compiling model...')
    model = get_unet()
    model_checkpoint = ModelCheckpoint(filepath='get_unet.hdf5', monitor='loss', save_best_only=True)

	# fit model
    print('Fitting model...')
    model.fit(trainData, trainMaskData, batch_size=batchSize, nb_epoch=numEpochs,  verbose=1, shuffle=True,
              callbacks=[model_checkpoint])

	# preprocess test data
    print('Loading and preprocessinging test data...')
    testData = load_test_data()
    testData = preprocess_data(testData)

	# init test data
    testData = testData.astype('float32')
    testData -= mean
    testData /= std

	# load saved weights
    print('Loading saved weights...')
    model.load_weights('get_unet.hdf5')

	# predict test data
    print('Predicting masks on test data...')
    testMaskData = model.predict(testData, verbose=1)
    np.save('testMaskData.npy', testMaskData) ##Predicting and saving the testMaskData as a vectorized npy file