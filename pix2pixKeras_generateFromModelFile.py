#Mostly from: https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
# example of pix2pix gan for satellite to map image-to-image translation
import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import os
import argparse
import shutil
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import skimage as skimage
from skimage import io


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y




# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=6, destDir=""):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	filename1 = os.path.join(destDir, filename1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	filename2 = os.path.join(destDir, filename2)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))



def makeFreshDir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

# load a single samples
def load_one_sample(filename, size=(256,256), syntheticSample=False):
	src_pixels = load_img(filename, target_size=size)
	# convert to numpy array
	src_img = img_to_array(src_pixels)
	if syntheticSample:
		src_img = getRidOfWeirdInaccuracies(src_img)
		src_img = enlargingAndPruningDots(src_img)
	src_img = (src_img - 127.5) / 127.5
	return src_img

# exporting the image from tkinter app using PIL seems to have made sometimes
# dodgey decisions in rounding. Let's smoothe them out
def getRidOfWeirdInaccuracies(img):
	ub = 200
	lb = 70
	img = np.where(img>(ub-1), 255, img)
	img = np.where(img<(lb+1), 0, img)
	img = np.where((img>lb)&(img<ub), 127.5, img)
	#print("Here are the unique values")
	#print(np.unique(img))
	return img


def createFileList(myDir, formats=['.tif', '.png']):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            for format in formats:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
    return fileList

def enlargingAndPruningDots_1(img):
	imgDots = img.copy()
	# for the dots, anything grey turns black
	imgDots = np.where(imgDots==127.5, 0, imgDots)
	mask = img.copy()
	# for the mask, anything white turns black
	mask = np.where(mask==255, 0, mask)
	greymask = mask.copy()
	mask = np.where(mask==127.5, 255, mask)
	#mask = np.where(mask!=127.5, 0, 255)
	dims = 3
	i = 16
	kernel = np.ones((dims,dims),np.uint8)
	kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
	imgDots = cv2.dilate(imgDots,kernel,iterations=i)
	#print(mask)
	imgDots = np.array(imgDots,  np.uint8)
	mask = 255 - mask
	mymask = np.array(mask,  np.uint8)
	imgDots = cv2.bitwise_and(imgDots, mymask)
	imgDots = imgDots + greymask
	return imgDots

def enlargingAndPruningDots(img):
	imgDots = img.copy()
	# for the dots, anything grey turns black
	imgDots = np.where(imgDots==127.5, 0, imgDots)
	mask = img.copy()
	# for the mask, anything white turns black
	mask = np.where(mask==255, 0, mask)
	greymask = mask.copy()
	mask = np.where(mask==127.5, 255, mask)
	#mask = np.where(mask!=127.5, 0, 255)
	dims = 3
	kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
	s1=10
	s2=100
	startingDots = countDots(imgDots, s1=10, s2=100)
	numDots = startingDots
	iterations = 1
	acceptableLostDots = 5
	if startingDots < 30:
		acceptableLostDots = 1
	while (numDots > (startingDots-acceptableLostDots)) & (iterations < 20):
	    #imgDots = cv2.dilate(imgDots,kernel,iterations=1)
	    #imgDots = cv2.morphologyEx(imgDots, cv2.MORPH_BLACKHAT, kernel)
	    dilation_size = 2
	    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
	                                   (dilation_size, dilation_size))
	    imgDots_new = cv2.dilate(imgDots, element)

	    numDots = countDots(imgDots_new, s1=s1, s2=s2)
	    if (numDots > (startingDots-5)):
	        imgDots = imgDots_new
	    #print("We've done {} iterations and there are {} dots".format(iterations, numDots))
	    iterations = iterations + 1
	    #s1 = s1*s1
	    s2 = s2*s2
	#print(mask)
	imgDots = np.array(imgDots,  np.uint8)
	mask = 255 - mask
	mymask = np.array(mask,  np.uint8)
	imgDots = cv2.bitwise_and(imgDots, mymask)
	imgDots = imgDots + greymask
	return imgDots

def countDots(img, s1=10, s2=100):
	#print(img.shape)
	## threshold
	grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	grayImage = np.array(grayImage,  np.uint8)
	#print(grayImage.shape)
	#cv2.imshow('gray image', grayImage)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	th, threshed = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

	## findcontours
	cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
	#im2, cnts, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	## filter by area
	#s1= 10
	#s2 = 100
	xcnts = []
	for cnt in cnts:
	    a = cv2.contourArea(cnt)
	    #print(a)
	    if s1<a <s2:
	        xcnts.append(cnt)

	#print("Dots number: {}".format(len(xcnts)))
	return len(xcnts)
	#Dots number: 23

def translate(model_file, source, dest, pheight=224, pwidth=224, syntheticSample=False):
	inputFilenames = createFileList(source)
	samples = []
	# creating the samples
	for name in inputFilenames:
		print("Sample name {}".format(name))
		s = load_one_sample(name, size=(256,256), syntheticSample=syntheticSample)
		samples.append(s)

	samples = np.asarray(samples)
	scaled_samples = (samples + 1) / 2.0
	print("The shape of the samples is: {}".format(samples.shape))

	# loading the model
	g_model = load_model(model_file)
	X_fakeB, _ = generate_fake_samples(g_model, samples, 1)
	X_fakeB = (X_fakeB + 1) / 2.0
	X_fakeB = np.uint8(X_fakeB*255)

	# Saving the stuff to a directory
	for i, name in enumerate(inputFilenames):
		bname = os.path.splitext(os.path.basename(name))[0]
		output_filename = "{}.png".format(bname)
		output_filename = os.path.join(dest, output_filename)
		print(output_filename)
		outImage = X_fakeB[i]
		resized_image = cv2.resize(outImage, (pwidth, pheight))
		#skimage.io.imsave(output_filename, outImage, check_contrast=False)
		skimage.io.imsave(output_filename, resized_image, check_contrast=False)



if __name__ == "__main__":
	model_inFiles_outFiles = ["models/aph_p2p_edToOri/", "mockUps", "mockUps_output", True, ""]
	model_inFiles_outFiles = ["models/aph_p2p_oriToEd/", "mockUps_real_short", "mockUps_real_output", False, "labelledPatches_short"]

	gen_models = createFileList(model_inFiles_outFiles[0], formats=['.h5'])
	#edToOri_model = "models/aph_p2p_edToOri/model_118800.h5"
	#inputFilenames = ["mockUps/trichome1.png", "mockUps/trichome2.png", "mockUps/trichome3.png"]
	inputFilenames = createFileList(model_inFiles_outFiles[1])
	outputDirBaseName = model_inFiles_outFiles[2]
	makeFreshDir(outputDirBaseName)
	ticSample = model_inFiles_outFiles[3]
	compareWithDir = model_inFiles_outFiles[4]

	samples = []
	for name in inputFilenames:
		print("Sample name {}".format(name))
		s = load_one_sample(name, size=(256,256), syntheticSample=syntheticSample)
		samples.append(s)

	samples = np.asarray(samples)
	scaled_samples = (samples + 1) / 2.0

	for testnum, gen_model in enumerate(gen_models):
		modelname = os.path.splitext(os.path.basename(gen_model))[0]
		g_model = load_model(gen_model)
		print("The shape of the samples is: {}".format(samples.shape))
		X_fakeB, _ = generate_fake_samples(g_model, samples, 1)
		X_fakeB = (X_fakeB + 1) / 2.0
		X_fakeB = np.uint8(X_fakeB*255)

		# Comparing with "ground truth"
		if compareWithDir != "":
			#Do the comparison
			for i, name in enumerate(inputFilenames):
				bname = os.path.split(os.path.basename(name))[1]
				dependentFileName = os.path.join(compareWithDir, bname)
				print("Comparing {} with {}".format(name,dependentFileName))


		# Saving the stuff to a directory
		for i, name in enumerate(inputFilenames):
			bname = os.path.splitext(os.path.basename(name))[0]
			output_filename = "{}_{}_fake.png".format(bname, modelname)
			output_filename = os.path.join(outputDirBaseName, output_filename)
			print(output_filename)
			outImage = X_fakeB[i]
			skimage.io.imsave(output_filename, outImage, check_contrast=False)




		# Now to display them...
		# This scaling is for pyplot?
		X_fakeB = (X_fakeB + 1) / 2.0
		n_samples = len(inputFilenames)
		n_samples = 1
		for i in range(n_samples):
			pyplot.subplot(3, n_samples, 1 + i)
			pyplot.axis('off')
			pyplot.imshow(scaled_samples[i])
		# plot generated target image
		for i in range(n_samples):
			pyplot.subplot(3, n_samples, 1 + n_samples + i)
			pyplot.axis('off')
			pyplot.imshow(X_fakeB[i])
		figName = "generatedImages_{}.png".format(testnum)
		pyplot.savefig(figName)
		pyplot.close()


	quit()
