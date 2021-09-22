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
import random
import patchRGB
import pix2pixKeras_generateFromModelFile as t #t for translation
import redDots

# Directory structure:
#

# Temporary directory names:
s_unlab_im          = "source_unlabelled_images" # rgb images, no labels, images
s_unlab_pat         = "source_unlabelled_patches" # rgb images, no labels, patches
s_lab_im            = "source_labelled_images" # rgb images, labels, reassembled
s_lab_pat           = "source_labelled_patches" # rgb images, labels, patches
s_lab_imr           = "source_labelled_reassembled" # rgb images, full pictures
s_dots_pat          = "source_dots_patches" # black and white images, dots, patches
s_dots_im           = "source_dots_images" # black and white images, dots, full images
s_dots_imr          = "source_dots_reassembled" # black and white images, cropped patches reassembled
s_bwg_imr           = "source_bwg_reassembled" # black, white grey images, cropped patches reassembled, predicted from the dots
t_bwg_pat           = "translated_bwg_patches" # black, white, grey images, patches
t_bwg_im            = "translated_bwg_reassembled" # black, white, grey images, image patches reassembled
visualise           = "compare_src_gt" # a visualisation of how our genrated images in t_bwg_im compares with s_bwg_imr



def create_temp_directories(dest_dir, dirs):
    # These are only kept if the destination directory is specified in the command line
    for dir in dirs:
        dirname = os.path.join(dest_dir, dir)
        os.makedirs(dirname)

def split_files_into_test_and_train_lists(srcList, testList, trainList, fraction):
    myFraction = 1/fraction
    for i, myfile in enumerate(srcList):
        if random.random() < myFraction:
            testList.append(myfile)
        else:
            trainList.append(myfile)


def copy_files_into_dir(fileList, dest_dir):
    for f in fileList:
        basename = os.path.basename(f)
        copyname = os.path.join(dest_dir, basename)
        shutil.copyfile(f, copyname)


if __name__ == "__main__":
    source_dir = "countCells_unlabel"
    dest_dir = "countCells_autogen"
    groundtruth_dir = "countCells_label"
    model_file = "models/aph_p2p_oriToEd/aph_p2p_oriToED/model_396000.h5" # A folder containing the actual models
    pheight = 224
    pwidth = 224
    keep_dest_dir = True

    # For Pam's Linux box
    #fullDatasetPath = "/home/pam/data/micropics/redDotDataset/redDotsSamples/redDotsSamples/aphanizemenon/"

    # For Pam's mac
    fullDatasetPath = "/Users/pam/Documents/data/micropics/"
    labelledDataPath = os.path.join(fullDatasetPath, "labels")
    unlabelledDataPath = os.path.join(fullDatasetPath, "originals")

    source_dir = unlabelledDataPath
    groundtruth_dir = labelledDataPath


    parser = argparse.ArgumentParser(description="Takes a folder of images and counts the cells in them")
    parser.add_argument("-s", "--source", help="the source directory where all the source images live")
    parser.add_argument("-g", "--groundtruth", help="the ground truth directory where all the already labelled images live")
    parser.add_argument("-d", "--dest",   help="the (optional) destination directory (will make it if it doesn't exist, wipe it if it does), stores intermediate images if specified")
    parser.add_argument("-m", "--modelfile", help="the file that contains the trained model (.h5)")
    parser.add_argument("-y", "--patchHeight", help="height of patches (pixels), default=224, actually defined by the model")
    parser.add_argument("-x", "--patchWidth", help="width of patches (pixels), default=224, actually defined by the model")

    args = parser.parse_args()

    if args.source:
        source_dir = args.source
    if args.groundtruth:
        groundtruth_dir = args.groundtruth
    if args.dest:
        dest_dir = args.dest
    if args.modelfile:
        model_file = args.modelfile
    if args.patchHeight:
        pheight = int(args.patchHeight)
    if args.patchWidth:
        pwidth = int(args.patchWidth)

    print("Getting pictures from {}".format(source_dir))
    print("Getting ground truth pictures from {}".format(groundtruth_dir))
    print("Storing intermediate pictures (and patches) to {} (will delete at end if not required).".format(dest_dir))
    print("Using the model in {} for translation".format(model_file))

    # Set up the directories
    patchRGB.makeFreshDir(dest_dir)
    # We should have test, train and models in that dir
    dirs = ["test", "train", "models"]
    create_temp_directories(dest_dir, dirs)
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    model_dir =  os.path.join(dest_dir, "model")
    # Both the training and test dir need all these
    dirs = [s_unlab_im, s_unlab_pat, s_lab_im, s_lab_pat, s_lab_imr, s_dots_pat, s_dots_im, s_dots_imr, s_bwg_imr, t_bwg_pat, t_bwg_im, visualise]
    create_temp_directories(train_dir, dirs)
    create_temp_directories(test_dir, dirs)
    test_src_label = os.path.join(test_dir, s_lab_im)
    test_src_unlabel = os.path.join(test_dir, s_unlab_im)
    train_src_label = os.path.join(train_dir, s_lab_im)
    train_src_unlabel = os.path.join(train_dir, s_unlab_im)


    # Sort out what is train and what is test:
    labelledFileNames = redDots.createFileList(labelledDataPath)
    testList = []
    trainList = []
    split_files_into_test_and_train_lists(labelledFileNames, testList, trainList, 10)

    copy_files_into_dir(testList, test_src_label)
    labelList = (x.replace("labels", "originals") for x in testList)
    copy_files_into_dir(labelList, test_src_unlabel)

    copy_files_into_dir(trainList, train_src_label)
    labelList = (x.replace("labels", "originals") for x in trainList)
    copy_files_into_dir(labelList, train_src_unlabel)

    # Now prep what we can without a model:
    dirs = ["train", "test"]
    for d in dirs:
        mydir = os.path.join(dest_dir, d)
        # Patch the labelled images
        dirname_src = os.path.join(mydir, s_lab_im)
        dirname_dst = os.path.join(mydir, s_lab_pat)
        patchRGB.patchDir(dirname_src, dirname_dst, pheight, pwidth, 0, 0)

        # Unpatch the labelled images (to account for crops)
        dirname_src = os.path.join(mydir, s_lab_pat)
        dirname_dst = os.path.join(mydir, s_lab_imr)
        patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)

        # Dots from ground truth (patches - gets the right image size)
        dirname_src = os.path.join(mydir, s_lab_pat)
        dirname_dst = os.path.join(mydir, s_dots_pat)
        redDots.getDotMaskDir(dirname_src, dirname_dst)

        # Unpatch the ground truth dot mask (to account for crops)
        dirname_src = os.path.join(mydir, s_dots_pat)
        dirname_dst = os.path.join(mydir, s_dots_imr)
        patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)

        # Enlarge the ground truth dots (for comparison)
        dirname_src = os.path.join(mydir, s_lab_imr)
        dirname_dst = os.path.join(mydir, s_bwg_imr)
        redDots.enlargeDotsDir(dirname_src, dirname_dst)

        # Dots from ground truth (images)
        dirname_dst = os.path.join(mydir, s_lab_im)
        dirname_dst = os.path.join(mydir, s_dots_im)
        redDots.getDotMaskDir(dirname_src, dirname_dst)

        # Patch the source
        dirname_src = os.path.join(mydir, s_unlab_im)
        dirname_dst = os.path.join(mydir, s_unlab_pat)
        patchRGB.patchDir(source_dir, dirname_dst, pheight, pwidth, 0, 0)

    quit()

    # Style transfer the source patches
    dirname_src = os.path.join(dest_dir, s_unlab_pat)
    dirname_dst = os.path.join(dest_dir, t_bwg_pat)
    t.translate(model_file, dirname_src, dirname_dst)

    # Tidy the style-transferred source patches
    #dirname_src = os.path.join(dest_dir, t_bwg_pat)
    #dirname_dst = os.path.join(dest_dir, t_bwg_pat)
    #redDots.tidyImageDir(dirname_src, dirname_dst)


    # Unpatch the translated image
    dirname_src = os.path.join(dest_dir, t_bwg_pat)
    dirname_dst = os.path.join(dest_dir, t_bwg_im)
    patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)

    # Tidy the style-transferred source complete image
    dirname_src = os.path.join(dest_dir, t_bwg_im)
    dirname_dst = os.path.join(dest_dir, t_bwg_im)
    redDots.tidyImageDir(dirname_src, dirname_dst, "")

    # Count the dots
    dirname_src = os.path.join(dest_dir, t_bwg_im)
    foundDots = redDots.countDotsDir(dirname_src)
    dirname_src = os.path.join(dest_dir, s_dots_imr)
    gtDots = redDots.countDotsDir(dirname_src)
    dirname_src = os.path.join(dest_dir, s_bwg_imr)
    gtBigDots = redDots.countDotsDir(dirname_src)

    for i in zip(foundDots, gtBigDots, gtDots):
        print("found cells {}; gt (processed): {}; gt (Ismael's) {}".format(i[0], i[1], i[2]))

    # lastly, visualise (somehow...)
    dirname = os.path.join(dest_dir, t_bwg_im)
    t_bwg_im_files = redDots.createFileList(dirname)
    dirname = os.path.join(dest_dir, s_dots_imr)
    s_dots_im_files = redDots.createFileList(dirname)
    dirname = os.path.join(dest_dir, s_bwg_imr)
    s_bwg_imr_files = redDots.createFileList(dirname)

    all_files = zip(t_bwg_im_files, s_dots_im_files)
    dirname_dst = os.path.join(dest_dir, visualise)
    for pair in all_files:
        img_mine = cv2.imread(pair[0])
        img_gt = cv2.imread(pair[1])
        #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)
        img_mine[img_mine>250]=170
        img_mine[img_gt>200]=255

        imageBaseName = os.path.splitext(os.path.basename(pair[0]))[0]
        save_name = os.path.join(dirname_dst, "{}.png".format(imageBaseName))
        skimage.io.imsave(save_name, img_mine, check_contrast=False)


    # tidy up
    if not keep_dest_dir:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
