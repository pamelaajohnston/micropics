import cv2 as cv2
import skimage as skimage
from skimage import io, data, color, filters
#from skimage.color import rgb2ycbcr
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from skimage.exposure import rescale_intensity

def showImage(image):
    plt.imshow(image, interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def findDotColour(img):
    print(img.shape)
    imgR = img.copy()
    imgR[:,:,1] = imgR[:,:,0]
    imgR[:,:,2] = imgR[:,:,2]
    #showImage(imgR)

    imgG = img.copy()
    imgG[:,:,0] = imgG[:,:,1]
    imgG[:,:,2] = imgG[:,:,1]
    #showImage(imgG)

    imgB = img.copy()
    imgB[:,:,0] = imgB[:,:,2]
    imgB[:,:,1] = imgB[:,:,2]
    #showImage(imgB)

    # Now to extract the red dots....
    #imgGDots = np.where(imgG < 25, 255, 0)
    #showImage(imgGDots)
    #imgRDots = np.where(imgR > 220, 255, 0)
    #showImage(imgRDots)

    indices = np.where(imgR > 220)
    print("The really red dots have RGU triple values of:")
    for i in range(0, len(indices[0])):
        myColour = img[indices[0][i], indices[1][i]]
        print(myColour)
    showImage(imgRDots)
    #imgDots = imgRDots | imgGDots

    indices = np.where(imgG < 25)
    print("The really NOT green dots have RGU triple values of:")
    for i in range(0, len(indices[0])):
        myColour = img[indices[0][i], indices[1][i]]
        print(myColour)
    showImage(imgRDots)

def getDotMask(img):
    #ismaelsRed = 230
    #ismaelsGreen = 27
    #ismaelsBlue = 35 # From using mac digital color meter in utilities
    ismaelsRed = 233
    ismaelsGreen = 22
    ismaelsBlue = 22 # From analysing the images
    ismaelsColour = [ismaelsBlue, ismaelsGreen, ismaelsRed]

    imgDotsChannel = np.where(np.all(img == ismaelsColour, axis=-1), 255, 0)
    imgDotsChannel = np.reshape(imgDotsChannel, (img.shape[0], img.shape[1], 1))
    imgDots = np.append(imgDotsChannel, imgDotsChannel, axis=2)
    imgDots = np.append(imgDots, imgDotsChannel, axis=2)

    i = imgDots.astype(np.uint8)
    #showImage(i)
    return i

def repeatChannelsx3(img_in):
    img_in = np.reshape(img_in, (img_in.shape[0], img_in.shape[1], 1))
    img_out = np.append(img_in, img_in, axis=2)
    img_out = np.append(img_out, img_in, axis=2)
    return img_out


def countDots(img):
    #print(img.shape)
    ## threshold
    grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th, threshed = cv2.threshold(grayImage, 100, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    #cv2.imshow('gray image', grayImage)
    #cv2.imshow('Thresholded image', threshed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    ## findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    #im2, cnts, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ## filter by area
    s1= 10
    s2 = 100
    xcnts = []
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        #print(a)
        if s1<a <s2:
            xcnts.append(cnt)

    print("Dots number: {}".format(len(xcnts)))
    #Dots number: 23

def hp_filter(img, binaryMask=False):
    #Greyscale it
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Normalise the greyscale
    img_norm = img_gray.copy()
    cv2.normalize(img_gray,  img_norm, 0, 255, cv2.NORM_MINMAX)
    img_float32 = np.float32(img_norm)

    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img_gray.shape
    crow, ccol = rows/2 , cols/2     # center

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 1

    mask2 = np.zeros((rows, cols, 2), np.uint8)
    mask2[np.where(mask == 0)] = 1
    #mask = mask2

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    myData = img_back.flatten()

    #threshold = 3.5e+08
    myMean = np.mean(myData)
    myStd = np.std(myData)
    myThres = myMean - (2*myStd)
    threshold = myThres
    backgroundPixel = 0
    trichomePixel = 255

    img_back = repeatChannelsx3(img_back)
    if (binaryMask):
        img_filtered = np.where(img_back>threshold, backgroundPixel, trichomePixel)
    else:
        img_filtered = np.where(img_back>threshold, backgroundPixel, img)

    return img_filtered

def grabCut(img, binaryMask=False):
    print(img.shape)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, img.shape[1] - 2, img.shape[0] - 2)


    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    if binaryMask:
        img = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    else:
        img = img * mask2[:, :, np.newaxis]
    return img

def grabCut2(img, binaryMask=False):
    # First, use the morphological filter to produce a mask2
    morphMask = morphFilter(img, True)
    properMask = prepGrabCutMask(morphMask)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask, bgModel, fgModel = cv2.grabCut(img, properMask, None, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    if binaryMask:
        img = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    else:
        img = img * mask2[:, :, np.newaxis]
    return img

def prepGrabCutMask(mask):
    #0 means background, 255 means foreground but it's through
    # draw over the lines to get a proper mask
    print("prepGrabCutMask")
    print(mask.shape)
    #showImage(mask)
    mask = mask[:,:,0]
    kernel = np.ones((7, 7))
    largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
    kernel = largeBlur
    properMask = mask.copy()
    myMask = convolve(mask, kernel)
    myMask2 = myMask.copy()
    myMask2 = 125
    myMask2 = np.where((myMask==0), 0, myMask2)
    myMask2 = np.where((mask==255), 255, myMask2)
    #showImage(repeatChannelsx3(myMask2))

    properMask = myMask2.copy()
    #properMask[myMask2 == 255] = 1
    #properMask[myMask2 == 125] = 0
    #properMask[myMask2 == 0] = 0
    properMask[myMask2 > 0] = cv2.GC_PR_FGD
    properMask[myMask2 == 255] = cv2.GC_FGD
    properMask[myMask2 == 0] = cv2.GC_BGD
    return np.uint8(properMask)


def morphFilter(img, binaryMask=False):
    #Greyscale it
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Normalise the greyscale
    #img_norm = img_gray.copy()
    #cv2.normalize(img_gray,  img_norm, 0, 255, cv2.NORM_MINMAX)
    gray = img_gray

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    sure_bg = repeatChannelsx3(sure_bg)
    backgroundPixel = 0
    trichomePixel = 255
    if binaryMask:
        img_filtered = np.where(sure_bg==0, backgroundPixel, trichomePixel)
    else:
        img_filtered = np.where(sure_bg==0, backgroundPixel, img)
    return img_filtered

# From https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
    # loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()
            # extra line in here fudging things up by Pam
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
    	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	# return the output image
	return output


def getTrichomeMask(img, binaryMask=False):
    #img_back = hp_filter(img, binaryMask)
    img_back = grabCut2(img, binaryMask)
    #img_back = morphFilter(img, binaryMask)
    return img_back





if __name__ == "__main__":
    imageNames = ["aphaniz_503.tiff", "aphaniz_558.tiff", "pabefore_17.png", "pabefore_1023.png", "pabefore_1396.png"]
    imageName = "aphaniz_503.tiff" # 503 has 45 dots by counting
    for imageName in imageNames:
        #img = skimage.io.imread(imageName)
        imageBaseName = os.path.splitext(imageName)[0]
        img_mat = cv2.imread(imageName)
        img = np.asarray(img_mat)
        height, width = img.shape[:2]
        print("Image {} is {} by {}".format(imageName, width, height))
        #showImage(img)
        #skimage.io.imsave("test.png", img)

        imgDots = getDotMask(img)
        countDots(imgDots)

        #showImage(imgDots)
        skimage.io.imsave("{}_dots.png".format(imageBaseName), imgDots, check_contrast=False)

        imgTrichome = getTrichomeMask(img_mat)
        # convert to rgb from bgr - why does opencv use BGR???
        imgTrichome_rgb = imgTrichome.copy()
        imgTrichome_rgb[:, :, 0] = imgTrichome[:, :, 2]
        imgTrichome_rgb[:, :, 2] = imgTrichome[:, :, 0]
        skimage.io.imsave("{}_trichome.png".format(imageBaseName), imgTrichome_rgb, check_contrast=False)
