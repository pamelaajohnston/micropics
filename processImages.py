import numpy as np
import os
import random
import shutil
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import metrics
from keras.backend import clear_session

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

from tabulate import tabulate
from PIL import Image

def modelArchitecture(num_classes, architectureNumber):
    print(num_classes)
    if architectureNumber == 0:
        modelName = "My first architecture"
        model = Sequential()
        model.add(Conv2D(32, (4, 4), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 1:
        modelName = "three 3x3 convs and fc"
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    return model, modelName

def createFileList(myDir, format='.png'):
    fileList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def processImage1(inFile, outDir):
    shutil.copy2(inFile, outDir)

def processImage2(inFile, outDir):
    im = Image.open(inFile)
    size = 128, 128
    img = im.resize(size, Image.LANCZOS)
    d, b = os.path.split(inFile)
    outName = os.path.join(outDir, b)
    print(outName)
    img.save(outName)

def processImage3(inFile, outDir):
    d, b = os.path.split(inFile)
    outName = os.path.join(outDir, b)
    im = cv2.imread(inFile, 0)
    im_norm = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
    size = 128, 128
    resized = cv2.resize(im_norm, size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(outName, resized)

def processImage(inFile, outDir):
    d, b = os.path.split(inFile)
    outName = os.path.join(outDir, b)
    im = cv2.imread(inFile, 0)
    im_norm = customFilter(im)
    size = 128, 128
    resized = cv2.resize(im_norm, size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(outName, resized)

def hp_filter(img):
    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
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
    return img_back

def customFilter(img):
    img_back = hp_filter(img)
    myData = img_back.flatten()
    #threshold = 3.5e+08
    myMean = np.mean(myData)
    myStd = np.std(myData)
    myThres = myMean - (2*myStd)
    threshold = myThres
    upper = 0
    img_filtered = np.where(img_back>threshold, upper, img)
    return img_filtered

def splitIntoTestTrain(src, dst):
    print(src)
    d, classname = os.path.split(src)
    fileList = createFileList(src)
    #10% randomness
    numSamples = len(fileList) // 10
    #if numSamples < 10:
    #    numSamples = 10
    shuffledFiles = fileList
    random.shuffle(shuffledFiles)
    print(shuffledFiles)
    test = shuffledFiles[0:numSamples]
    train = shuffledFiles[numSamples:]

    testdir = "{}/test/{}".format(dst, classname)
    os.makedirs(testdir)
    traindir = "{}/train/{}".format(dst, classname)
    os.makedirs(traindir)
    
    for myFile in test:
        processImage(myFile, testdir)
    
    for myFile in train:
        processImage(myFile, traindir)
    return len(test), len(train)


if __name__ == "__main__":
    batch_size = 64
    epochs = 15
    img_w, img_h = 128, 128

    print("\n\n\n")

    num_classes = 2
    srcpics0 = "/home/pam/data/micropics/before"
    srcpics1 = "/home/pam/data/micropics/after"
    datadir = "/home/pam/data/micropics/workingSet"
    trainSamples = 0
    testSamples = 0
    valSamples = 0

    testdir = "{}/test".format(datadir)
    traindir = "{}/train".format(datadir)
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    os.makedirs(testdir)
    if os.path.exists(traindir):
        shutil.rmtree(traindir)
    os.makedirs(traindir)

    te, tr = splitIntoTestTrain(srcpics0, datadir)
    trainSamples += tr
    testSamples += te
    te, tr = splitIntoTestTrain(srcpics1, datadir)
    trainSamples += tr
    testSamples += te
 

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 3)

    # Note that there isn't any data augmentation
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip = True,
        zca_whitening = True,
        rotation_range = 180
    )
    train_it = datagen.flow_from_directory('{}/train'.format(datadir),
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=64,
                                           class_mode="categorical",
                                           shuffle=True)

    test_it = datagen.flow_from_directory('{}/test'.format(datadir),
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=64,
                                           class_mode="categorical",
                                           shuffle=False)


    # And now the model....
    modelArchitectures = range(0, 7)
    modelArchitectures = [1,]
    epochNumbers = [3, 5, 10]
    epochNumbers = [10, 25, 30]
    optimisers = ["rmsprop", "sgd", "adam", "adagrad"]
    optimisers = ["sgd", "adam"]

    bestf1 = 0
    bestNetwork = "unknown"
    bestMCC = 0
    bestAccuracy = 0
    bestf1model = 0
    bestMCCmodel = 0
    bestAccuracyModel = 0


    resultsList = []
    #resultsList.append(("archNo", "epochs", "opt", "f1", "acc", "mcc"))
    for architectureNumber in modelArchitectures:
        for epochs in epochNumbers:
            for optimiser in optimisers:
                model, modelName = modelArchitecture(num_classes, architectureNumber)


                print("Compiling the model: {}".format(modelName))
                model.compile(loss='mse',
                              optimizer=optimiser,
                              metrics=[metrics.categorical_accuracy])


                stepsPerEpoch = trainSamples // batch_size
                if stepsPerEpoch < 20:
                    stepsPerEpoch = 20
                print(stepsPerEpoch)
                


                valSteps = valSamples // batch_size

                print("Fitting the model: {}".format(modelName))
                model.fit_generator(
                    train_it,
                    steps_per_epoch=stepsPerEpoch,
                    epochs=epochs,
                    validation_data=test_it,
                    validation_steps=valSteps)

                probabilities = model.predict_generator(generator=test_it)
                #print(probabilities)
                y_pred = np.argmax(probabilities, axis=-1)
                #print(y_pred)
                y_true = test_it.classes
                #print(y_true)

                cm = confusion_matrix(y_true, y_pred)
                print("The stats for {} after {} epochs with {} opt:".format(modelName, epochs, optimiser))
                f1 = f1_score(y_true, y_pred, average='micro')
                f1_all = f1_score(y_true, y_pred, average=None)
                mcc = matthews_corrcoef(y_true, y_pred)
                acc = accuracy_score(y_true, y_pred, normalize=True)
                print(cm)
                print("f1 micro = {} and all {} ".format(f1, f1_all))
                print("accuracy = {}".format(acc))
                print("mcc = {}".format(mcc))
                myResults = (architectureNumber, epochs, optimiser, f1, acc, mcc)
                resultsList.append(myResults)
                # print out the results as we go...
                print(tabulate(resultsList, headers=["archNo", "epochs", "opt", "f1", "acc", "mcc"]))

                saveModel = False

                if f1 > bestf1:
                    bestf1 = f1
                    bestf1model = model
                    saveModel = True

                if mcc > bestMCC:
                    bestMCC = mcc
                    bestMCCmodel = model
                    saveModel = True

                if acc > bestAccuracy:
                    bestAccuracy = acc
                    bestAccuracyModel = model
                    saveModel = True

                # save model to file
                if saveModel:
                    modelBaseFilename = "arch{}_epochs{}_opt{}".format(architectureNumber, epochs, optimiser)
                    print("Saving to {}".format(modelBaseFilename))
                    model_json = model.to_json()
                    with open("{}.json".format(modelBaseFilename), "w") as json_file:
                        json_file.write(model_json)
                    model.save_weights("{}.h5".format(modelBaseFilename))
                clear_session()

    print("The overall results:")
    print(resultsList)

