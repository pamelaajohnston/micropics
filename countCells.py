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
import patchRGB
import pix2pixKeras_generateFromModelFile as t #t for translation
import redDots

# Temporary directory names:
src_pat         = "source_patches"
src_pat_trans   = "source_patches_translated"
src_trans       = "source_translated"
gt_pat          = "groundtruth_patches"
gt_pat_trans    = "groundtruth_patches_dots"
gt_trans        = "groundtruth_translated"
gt_dots         = "groundtruth_dots"

def create_temp_directories(dest_dir):
    # These are only kept if the destination directory is specified in the command line
    dirs = [src_pat, gt_pat, src_pat_trans, gt_pat_trans, src_trans, gt_dots]
    for dir in dirs:
        dirname = os.path.join(dest_dir, dir)
        os.makedirs(dirname)


if __name__ == "__main__":
    source_dir = "countCells_unlabel"
    dest_dir = "countCells_autogen"
    groundtruth_dir = "countCells_label"
    model_file = "models/aph_p2p_oriToEd/aph_p2p_oriToED/model_396000.h5" # A folder containing the actual models
    pheight = 224
    pwidth = 224
    keep_dest_dir = True

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
    create_temp_directories(dest_dir)

    # Patch the ground truth
    dirname = os.path.join(dest_dir, gt_pat)
    patchRGB.patchDir(groundtruth_dir, dirname, pheight, pwidth, 0, 0)

    # Unpatch the ground truth (to account for crops)
    dirname_src = os.path.join(dest_dir, gt_pat)
    dirname_dst = os.path.join(dest_dir, gt_trans)
    patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)

    # Dots from ground truth (patches)
    dirname_src = os.path.join(dest_dir, gt_pat)
    dirname_dst = os.path.join(dest_dir, gt_pat_trans)
    redDots.getDotMaskDir(dirname_src, dirname_dst)

    # Dots from ground truth (images)
    dirname = os.path.join(dest_dir, gt_dots)
    redDots.getDotMaskDir(groundtruth_dir, dirname)

    # Patch the source
    dirname = os.path.join(dest_dir, src_pat)
    patchRGB.patchDir(source_dir, dirname, pheight, pwidth, 0, 0)

    # Translate the source patches
    dirname_src = os.path.join(dest_dir, src_pat)
    dirname_dst = os.path.join(dest_dir, src_pat_trans)
    t.translate(model_file, dirname_src, dirname_dst)

    # Unpatch the translated image
    dirname_src = os.path.join(dest_dir, src_pat_trans)
    dirname_dst = os.path.join(dest_dir, src_trans)
    patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)


    # tidy up
    if not keep_dest_dir:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
