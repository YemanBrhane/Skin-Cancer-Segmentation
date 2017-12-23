#=======================================================================
#
# This is the file for converting vectorized file to image
#
#=======================================================================

#-----------------------------------------------------------------------
# Libs	
#-----------------------------------------------------------------------
import numpy as np
from scipy.misc import toimage, imsave
import os



#-----------------------------------------------------------------------
# Variables
#-----------------------------------------------------------------------
# directory for images
imageDirInput = "imgs_mask_test_01"
imageDir = './' + imageDirInput + '.npy'
imagesSaved = np.load(imageDir)
imageDirOutput = 'results\testMaskData'
# these are the selected 'random' 5 images to test
# names will be matched by index for now: 1=22, 2=234...
list_of_random_index = np.arange(379)



#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
# output directory
def ensure_directory_exist(directory_name):
    isExist = os.path.isdir('./' + directory_name)
    if not isExist:
        os.mkdir(directory_name)


# use this to show the image rather than save it, if you want
def show_image(image):
    toimage(image).show()


# save the image array to a .png file
def plot_image_save_to_file(name, img_cur):
    #  ensure a directory is present/build if necessary
    saveDir = imageDirOutput  # from global value
    ensure_directory_exist(saveDir)
	
    #  build full path and save
    fileName = name + '.png'
    fullPathName = os.path.join(saveDir, fileName)
    imsave(fullPathName, img_cur)


# convert the numpy array to a int array through .astype('float32')
def convert_numpy_array_to_int_array(imagesSaved):
	# print number of pictures
    print(len(imagesSaved))   
    imageList = []
    i = 0
    while i < len(imagesSaved):
        for photo_indiv in imagesSaved[i]:
            image = photo_indiv.astype('float32')
            imageList.append(image)
            # plot_image_save_to_file("jack", image)
            # print(image)
        i += 1
    return imageList


# loop through converted int array and save the image to .png
def convert_int_array_to_png(imageList):
    imageId = 1
    for photo_array in imageList:
        name = imageDirInput + '_' + str(imageId)
        plot_image_save_to_file(name, photo_array)
        imageId += 1


# create a list of 5 int image array
def get_random_5(imageArrayInteger):
    mySet = set()
    smallList = []
    for selected_index in list_of_random_index:
        mySet.add(selected_index)
    i = 0
    while i < len(imageArrayInteger):
        if i in mySet:
            smallList.append(imageArrayInteger[i])
        i += 1
    return smallList


# wrapper to create 5 'random'(spec. gloablly) .png files to view binary mask
def convert_random_5(imageArrayInteger):
    smallList = get_random_5(imageArrayInteger)
    convert_int_array_to_png(smallList)


	
#-----------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------
def main():
    imageArrayInteger = convert_numpy_array_to_int_array(imagesSaved)
    convert_random_5(imageArrayInteger)  # TODO: make sure naming matches
	
main()