#=======================================================================
#
# This is the file for converting image to vectorized file
#
#=======================================================================

#-----------------------------------------------------------------------
# Libs	
#-----------------------------------------------------------------------
from __future__ import print_function
import os
import numpy as np
import skimage
from skimage import data, draw
from skimage import transform, util
import cv2



#-----------------------------------------------------------------------
# Variables
#-----------------------------------------------------------------------
# directory for images
dataPath = 'D:\MAIA_Italy\SKIN_CANCER'

# image size for resizing
numRows = 64*3
numCols = 80*3



#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
# convert train data to vectorized file
def create_train_data():
	# init
    trainDataPath = os.path.join(dataPath, 'train')
    images = os.listdir(trainDataPath)
    total = len(images)
    imagesSaved = np.ndarray((total, 1, numRows, numCols), dtype=np.uint8)
    i = 0
    print('Creating train images...')
	# for each image: read, rezie and convert 
    for imageName in images:
        print(imageName)
        # read RGB image
        image = cv2.imread(os.path.join(trainDataPath, imageName))
        # convert to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # rescale to numRows x numCols size
        imageResized = cv2.resize(imageGray, (numCols,numRows))
        image = np.array([imageResized])
        imagesSaved[i] = image
        print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
	# save file
    np.save('trainData.npy', imagesSaved)
    print('Saving to .npy files done.')

	
# convert mask data to vectorized file
def create_train_masks_data():
	# init
    trainMasksDataPath = os.path.join(dataPath, 'train_masks')
    images = os.listdir(trainMasksDataPath)
    total = len(images)
    imagesSaved = np.ndarray((total, 1, numRows, numCols), dtype=np.uint8)
    i = 0
	# for each image: read, rezie and convert 
    print('Creating train masks images...')
    for imageName in images:
        print(imageName)
        # read RGB image
        image = cv2.imread(os.path.join(trainMasksDataPath, imageName))
        # convert to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # fRescale to numRows x numCols size
        imageResized = cv2.resize(imageGray, (numCols,numRows))
        image = np.array([imageResized])
        image = np.array([image])
        imagesSaved[i] = image
        print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
	# save file
    np.save('trainMaskData.npy', imagesSaved)
    print('Saving to .npy files done.')


# convert mask data to vectorized file
def create_test_data():
	# init
    testDataPath = os.path.join(dataPath, 'test')
    images = os.listdir(testDataPath)
    total = len(images)
    imagesSaved = np.ndarray((total, 1, numRows, numCols), dtype=np.uint8)
    imagesSavedId = np.ndarray((total, ), dtype=np.int32)
    i = 0
    print('Creating test images...')
	# for each image: read, rezie and convert 
    for imageName in images:
        print(imageName)
        # read RGB image
        image = cv2.imread(os.path.join(testDataPath, imageName))
        # convert to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # rescale to numRows x numCols size
        imageResized = cv2.resize(imageGray, (numCols,numRows))
        image = np.array([imageResized])
        image = np.array([image])
        imagesSaved[i] = image
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
	# save file
    np.save('testData.npy', imagesSaved)
    print('Saving to .npy files done.')



#-----------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------
if __name__ == '__main__':
    create_train_data()
    create_train_masks_data()
    create_test_data()