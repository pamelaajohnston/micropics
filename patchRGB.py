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


def makeFreshDir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

if __name__ == "__main__":
    pwidth = 224
    pheight = 224
    hstride = 0
    vstride = 0

    parser = argparse.ArgumentParser(description="Takes a folder of images and creates patches from them")
    parser.add_argument("-s", "--source", help="the source directory")
    parser.add_argument("-d", "--dest",   help="the destination directory (will make it if it doesn't exist, wipe it if it does)")
    parser.add_argument("-y", "--patchHeight", help="height of patches (pixels), default=224")
    parser.add_argument("-x", "--patchWidth", help="width of patches (pixels), default=224")
    parser.add_argument("-v", "--vertStride", help="Number of pixels between patches, vertically (default=0)")
    parser.add_argument("-z", "--horzStride", help="Number of pixels between patches, horizontally (default=0)")

    args = parser.parse_args()

    if args.source:
        print("Getting pictures from {}".format(args.source))
        imageNames = createFileList(args.source)
        print("*****************************************")
        print("Getting patches from the following files:")
        print(imageNames)
        print("*****************************************")
    if args.dest:
        destDir = args.dest
        print("Storing patch pictures to {}.".format(args.dest))
        makeFreshDir(destDir)
    if args.patchHeight:
        pheight = int(args.patchHeight)
    if args.patchWidth:
        pwidth = int(args.patchWidth)
    if args.vertStride:
        vstride = int(args.vertStride)
    if args.horzStride:
        hstride = int(args.horzStride)

    for imageName in imageNames:
        imageBaseName = os.path.splitext(os.path.basename(imageName))[0]
        print(imageBaseName)
        img_mat = cv2.imread(imageName)
        img = np.asarray(img_mat)
        iheight, iwidth = img.shape[:2]
        print("Image {} is {} by {}".format(imageName, iwidth, iheight))
        vstart = 0
        vend = iheight - (pheight) + 1
        hstart = 0
        hend = iwidth - (pwidth) + 1

        print("Starting at {}, last patch from {}".format(vstart, vend))
        print(range(vstart, vend, (pheight+vstride)))
        print("Starting at {}, last patch from {}".format(hstart, hend))
        print(range(hstart, hend, (pwidth+hstride)))

        for j in range(vstart, vend, (pheight+vstride)):
            for i in range(hstart, hend, (pwidth+hstride)):
                crop_img = img[j:j+pheight, i:i+pwidth]
                cropName = os.path.join(destDir, "{}_patch{}x{}.png".format(imageBaseName, i, j))
                skimage.io.imsave(cropName, crop_img, check_contrast=False)
