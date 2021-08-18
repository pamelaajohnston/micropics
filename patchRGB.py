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

def patchDir(source, dest, pheight, pwidth, vstride, hstride):
    imageNames = createFileList(source)
    #print("*****************************************")
    #print("Getting patches from the following files:")
    #print(imageNames)
    #print("*****************************************")
    print("Patching the directory {} into {}".format(source, dest))
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
                cropName = os.path.join(dest, "{}_patch{}x{}.png".format(imageBaseName, i, j))
                #skimage.io.imsave(cropName, crop_img, check_contrast=False)
                cv2.imwrite(cropName, crop_img)

def unpatchDir(source, dest, pheight, pwidth):
    imageNames = createFileList(source)
    imageNames_b = [os.path.splitext(os.path.basename(i))[0] for i in imageNames]
    images = [i.split("_patch")[0] for i in imageNames_b]
    images = list(set(images))
    print("The images in the directory are: {}".format(images))
    for image in images:
        patchNames_full = [k for k in imageNames if image in k]
        patchNames0 = [k for k in imageNames_b if image in k]
        patchNames = [k.replace(".png", "") for k in imageNames_b if image in k]
        patchdims = [i.split("_patch")[1] for i in patchNames]
        #print(patchdims)
        patchx = [int(i.split("x")[0]) for i in patchdims]
        patchy = [int(i.split("x")[1]) for i in patchdims]
        width = max(patchx) + pwidth
        height = max(patchy) + pheight
        print("The patches for {} will make a {} by {} image".format(image, width, height))

        size = (height, width, 3)
        m = np.zeros(size, dtype=np.uint8)
        #print(patchNames_full)
        for patch in patchNames_full:
            img_mat = cv2.imread(patch)
            p = np.asarray(img_mat)
            patchName = os.path.splitext(os.path.basename(patch))[0]
            patchName = patchName.replace(image, "")
            patchName = patchName.replace("_patch", "")
            xco = int(patchName.split("x")[0])
            yco = int(patchName.split("x")[1])
            m[yco:yco+pheight, xco:xco+pwidth] = p
        imageName = os.path.join(dest, "{}.png".format(image))
        #skimage.io.imsave(cropName, crop_img, check_contrast=False)
        cv2.imwrite(imageName, m)

        #patchx = list(set(patchx)).sort()
        #patchy = list(set(patchy)).sort()
        #print(patchx)
        #print(patchy)

        #print(patchNames)
        #patchNames0 = patchNames0.sort()
        #print(patchNames0)

        #for y in range(0, max(patchy)):
        #    for x in range(0, max(patchx)):



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
        source = args.source
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

    patchDir(source, destDir, pheight, pwidth, vstride, hstride)
