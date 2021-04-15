import cv2 as cv2
import skimage as skimage
from skimage import io, data, color, filters
#from skimage.color import rgb2ycbcr
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from skimage.exposure import rescale_intensity
import argparse
import shutil

def createFileList(myDir, formats=['.tif', '.png', '.tiff']):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            for format in formats:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
    return fileList

def isItBlank(img):
    #Here's the place to customise. A blank file here is where every value in the array is the same
    numUniques = len(np.unique(img))
    if (numUniques) < 2:
        return True
    else:
        return False

if __name__ == "__main__":
    pwidth = 224
    pheight = 224
    hstride = 0
    vstride = 0

    parser = argparse.ArgumentParser(description="Takes a folder of images and creates patches from them")
    parser.add_argument("-b", "--blankPatches", help="the source directory containing the patches that will be checked")
    parser.add_argument("-c", "--correspondingPatches",   help="the source directory that will also have files removed")

    args = parser.parse_args()

    if args.blankPatches:
        print("Getting pictures from {}".format(args.blankPatches))
        imageNames = createFileList(args.blankPatches)
        print("*****************************************")
        print("Getting patches from the following files:")
        print(imageNames)
        print("*****************************************")
    if args.correspondingPatches:
        destDir = args.correspondingPatches
        print("Dependent folder that will also be modified {}.".format(args.correspondingPatches))

    for imageName in imageNames:
        imageBaseName = os.path.split(os.path.basename(imageName))[1]
        dependentFileName = os.path.join(destDir, imageBaseName)
        print(imageName)
        print(dependentFileName)
        img_mat = cv2.imread(imageName)
        img = np.asarray(img_mat)
        iheight, iwidth = img.shape[:2]
        print("Image {} is {} by {}".format(imageName, iwidth, iheight))

        blank = isItBlank(img)
        if blank:
            #remove the file and the corresponding files
            os.remove(imageName)
            os.remove(dependentFileName)
