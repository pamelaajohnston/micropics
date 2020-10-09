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

def sortOutTwoPoints(x0, y0, x1, y1):
    myForwardTuple = (x0, y0, x1, y1)
    myReverseTuple = (x1, y1, x0, y0)
    myTuple = myForwardTuple
    if x1 < x0:
        myTuple = myReverseTuple
    if x0 == x1:
        if y1 < y0:
            myTuple = myReverseTuple
    return myTuple

def findRightAndDownNeighbour(ax, ay, idx, listb):
    bx = listb[idx, 1]
    by = listb[idx, 2]

    while (bx < ax) & (by < ay):
        idx = idx + 1
        if idx > len(listb):
            bx = ax
            by = ay
        else:
            bx = listb[idx, 1]
            by = listb[idx, 2]
    return (ax, ay, bx, by)

def joinDots(img):
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
        if s1 < a < s2:
            xcnts.append(cnt)

    # from https://stackoverflow.com/questions/51022381/how-do-i-connect-closest-points-together-using-opencv
    listx = []
    listy = []
    for cnt in xcnts:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        listx.append(cX)
        listy.append(cY)
    listxy = list(zip(listx,listy))
    listxy = np.array(listxy)
    #print("listxy:")
    #print(listxy)

    # for every (x,y) in listxy, find the nearest neighbours and connect to them
    distanceList = [] # This will be filled with point a, point b, distance
    connections = []
    onePoint = False
    if onePoint:
        for a in listxy:
            aListDist = []
            xlist = []
            ylist = []
            for b in listxy:
                dist = np.linalg.norm(a-b)
                #print("Distance is {}".format(dist))
                if dist == 0:
                    pass
                else:
                    myTuple = [a, b, dist]
                    distanceList.append(myTuple)
                    aListDist.append(dist)
                    xlist.append(b[0])
                    ylist.append(b[1])
            aDists = list(zip(aListDist, xlist, ylist))
            sort = sorted(aDists, key=lambda second: second[0])
            sort = np.array(sort)
            # check for duplications and go for next nearest neighbour if there is a duplication
            idx = 0
            b = int(sort[idx,1]), int(sort[idx,2])
            # Let's always organise the tuple from left to right, top to bottom
            myTuple = sortOutTwoPoints(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            #myTuple = findRightAndDownNeighbour(a[0], a[1], idx, sort)
            print("Joining a({}, {}) and b({}, {}) distance {}".format(myTuple[0], myTuple[1], myTuple[2], myTuple[3], sort[0,0]))
            while (myTuple in connections):
                idx = idx + 1
                if idx < len(sort):
                    myTuple = sortOutTwoPoints(int(a[0]), int(a[1]), int(sort[idx,1]), int(sort[idx,2]))
                    #myTuple = findRightAndDownNeighbour(a[0], a[1], idx, sort)
                    print("Alternative: Joining a({}, {}) and b({}, {}) distance {}".format(myTuple[0], myTuple[1], myTuple[2], myTuple[3], sort[idx,0]))
                else:
                    myTuple = null
                    print("No other point found?")
            if myTuple:
                connections.append(myTuple)
        else: # join the nearest two neighbours that form the most straight line with a
        for a in listxy:
            aListDist = []
            xlist = []
            ylist = []
            for b in listxy:
                dist = np.linalg.norm(a-b)
                #print("Distance is {}".format(dist))
                if dist == 0:
                    pass
                else:
                    myTuple = [a, b, dist]
                    distanceList.append(myTuple)
                    aListDist.append(dist)
                    xlist.append(b[0])
                    ylist.append(b[1])
            aDists = list(zip(aListDist, xlist, ylist))
            sort = sorted(aDists, key=lambda second: second[0])
            sort = np.array(sort)
            # check for duplications and go for next nearest neighbour if there is a duplication
            idx = 0
            maxidx = 5 # only sort through the top 5 nearest neighbours, I think.
            pairsList = []
            for i in range(0, maxidx):
                for j in range(0, maxidx):
                    if i != j:
                        pairsList.append((i, j))
            minDiff = 1920
            c1 = a
            c2 = a
            for pointPair in pairsList:
                idx1 = pointPair[0]
                idx2 = pointPair[1]
                b1 = int(sort[idx1,1]), int(sort[idx1,2])
                dist_atob1 = sort[idx1, 0]
                b2 = int(sort[idx2,1]), int(sort[idx2,2])
                dist_atob2 = sort[idx2, 0]
                dist_b1tob2 = np.linalg.norm(b1-b2)
                if dist_b1tob2 != 0:
                    diff = abs((dist_atob1 + dist_atob2) - dist_b1tob2)
                    if diff < minDiff:
                        minDiff = diff
                        c1 = b1
                        c2 = b2

            myTuple = sortOutTwoPoints(int(a[0]), int(a[1]), int(c1[0]), int(c1[1]))
            connections.append(myTuple)
            myTuple = sortOutTwoPoints(int(a[0]), int(a[1]), int(c2[0]), int(c2[1]))
            connections.append(myTuple)

    #connections = np.array(connections)
    if len(connections) != len(set(connections)):
        print("There were some duplicates in the list: {} vs ".format(len(connections), len(set(connections))))
    for cn in connections:
        cv2.line(img, (cn[0],cn[1]), (int(cn[2]), int(cn[3])), (0,0,255), 2)




    print("Dots number: {}".format(len(xcnts)))
    #Dots number: 23
    return img


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
        img = repeatChannelsx3(img)
    else:
        img = img * mask2[:, :, np.newaxis]
    return img

def grabCut2(img, binaryMask=False):
    # First, use the morphological filter to produce a mask2
    morphMask = morphFilter(img, True, 16)
    properMask = prepGrabCutMask(morphMask)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask, bgModel, fgModel = cv2.grabCut(img, properMask, None, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    if binaryMask:
        img = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        img = repeatChannelsx3(img)
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
    properMask[myMask2 > 0] = cv2.GC_PR_FGD
    properMask[myMask2 == 255] = cv2.GC_FGD
    properMask[myMask2 == 0] = cv2.GC_BGD
    return np.uint8(properMask)


def morphFilter(img, binaryMask=False, dims=3):
    #Greyscale it
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Normalise the greyscale
    #img_norm = img_gray.copy()
    #cv2.normalize(img_gray,  img_norm, 0, 255, cv2.NORM_MINMAX)
    gray = img_gray

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((dims,dims),np.uint8)
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
    #img_back = morphFilter(img, binaryMask, 3)
    return img_back

def shapeDetection(img, bin_1c):
    print(img.shape)
    doHough = False
    if doHough:
        edges = cv2.Canny(bin_1c, 50, 200)
        # Detect points that form a line
        minLineLength = 10
        maxLineGap = 25
        lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
        # Draw lines on the image
        for line in lines:
            print("A line!")
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        return img

    dist = cv2.distanceTransform(bin_1c, cv2.DIST_L2, 3)
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    #cv2.imshow('Distance Transform Image', dist)
    #showImage(repeatChannelsx3(dist))





if __name__ == "__main__":
    imageNames = ["aphaniz_503.tiff", "aphaniz_558.tiff", "pabefore_17.png", "pabefore_1023.png", "pabefore_1396.png"]
    imageNames = ["pabefore_1396.png"] # 503 has 45 dots by counting
    imageNames = ["aphaniz_558.tiff"] # 503 has 45 dots by counting
    countingDots = False
    joiningDots = True
    gettingTrichomes = False
    for imageName in imageNames:
        #img = skimage.io.imread(imageName)
        imageBaseName = os.path.splitext(imageName)[0]
        img_mat = cv2.imread(imageName)
        img = np.asarray(img_mat)
        height, width = img.shape[:2]
        print("Image {} is {} by {}".format(imageName, width, height))
        #showImage(img)
        #skimage.io.imsave("test.png", img)

        if countingDots:
            imgDots = getDotMask(img)
            countDots(imgDots)
            #showImage(imgDots)
            skimage.io.imsave("{}_dots.png".format(imageBaseName), imgDots, check_contrast=False)
        if joiningDots:
            imgDots = getDotMask(img)
            imgDots = joinDots(imgDots)
            #showImage(imgDots)
            skimage.io.imsave("{}_dotsjoined.png".format(imageBaseName), imgDots, check_contrast=False)

        if gettingTrichomes:
            binaryMask = True
            imgTrichome = getTrichomeMask(img_mat, binaryMask=binaryMask)
            # convert to rgb from bgr - why does opencv use BGR???
            imgTrichome_rgb = imgTrichome.copy()
            imgTrichome_rgb[:, :, 0] = imgTrichome[:, :, 2]
            imgTrichome_rgb[:, :, 2] = imgTrichome[:, :, 0]
            skimage.io.imsave("{}_trichome.png".format(imageBaseName), imgTrichome_rgb, check_contrast=False)
            bin_1c = imgTrichome[:, :, 0]
            img_lines = shapeDetection(img_mat, bin_1c)
            skimage.io.imsave("{}_houghlines.png".format(imageBaseName), img_lines, check_contrast=False)
