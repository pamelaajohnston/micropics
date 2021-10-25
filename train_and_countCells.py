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
import generatePictures
import pix2pixKeras as m # m for model
import pandas as pd

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
s_bwg_pat           = "source_bwg_patches" # black, white grey images, cropped patches reassembled, predicted from the dots
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
    do_test_train_split = False
    set_up_files = True
    evaluate_enlarge_dots = True
    create_model = False
    run_models = False


    # For Pam's Linux box
    fullDatasetPath = "/home/pam/data/micropics/redDotDataset/redDotsSamples/redDotsSamples/aphanizomenon/"
    myoutput = open("train_and_countCells_results.txt",'w')

    # For Pam's mac
    #fullDatasetPath = "/Users/pam/Documents/data/micropics/aphaniz"
    #fullDatasetPath = "/Users/pam/Documents/data/micropics/smallSet"


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

    print("Getting pictures from {}".format(source_dir), file=myoutput)
    print("Getting ground truth pictures from {}".format(groundtruth_dir), file=myoutput)
    print("Storing intermediate pictures (and patches) to {} (will delete at end if not required).".format(dest_dir), file=myoutput)
    print("Getting pictures from {}".format(source_dir))
    print("Getting ground truth pictures from {}".format(groundtruth_dir))
    print("Storing intermediate pictures (and patches) to {} (will delete at end if not required).".format(dest_dir))

    #model_name, do_test_train_split, set_up_files, create_model, run_models, patch_dim, batch_size, big_dots_type, trichome_type
    parameters_to_change = [
        ["base1_224",                   False,  False,  False,  True, 224, 1, "trichome_on_top", "hp_filter"],
        #["base1_224",                  True,   True,   True,   True, 224, 1, "trichome_on_top", "hp_filter"],
        #["base1_224_no_trichome_5",    False,  True,   True,   True, 224, 1, "big_dots_only", "hp_filter"],
        #["base1_224_no_trichome_2",    False,  True,   True,   True, 224, 1, "big_dots_only_error_2", "hp_filter"],
        ["base1_224_morph",             False,  True,   True,   True, 224, 1, "trichome_on_top", "morph_filter"],
        ["base1_224_grabcut",           False,  True,   True,   True, 224, 1, "trichome_on_top", "grabCut"],

        ["base10_224",                  False,  True,   True,   True, 224, 10, "trichome_on_top", "hp_filter"],
        ["base10_224_no_trichome_5",    False,  True,   True,   True, 224, 10, "big_dots_only", "hp_filter"],
        ["base10_224_no_trichome_2",    False,  True,   True,   True, 224, 10, "big_dots_only_error_2", "hp_filter"],
        ["base10_224_morph",            False,  True,   True,   True, 224, 10, "trichome_on_top", "morph_filter"],
        ["base10_224_grabcut",          False,  True,   True,   True, 224, 10, "trichome_on_top", "grabCut"],

        ["base1_128",                   False,  True,   True,   True, 128, 1, "trichome_on_top", "hp_filter"],
        ["base1_128_no_trichome_5",     False,  True,   True,   True, 128, 1, "big_dots_only", "hp_filter"],
        ["base1_128_no_trichome_2",     False,  True,   True,   True, 128, 1, "big_dots_only_error_2", "hp_filter"],
        ["base1_128_morph",             False,  True,   True,   True, 128, 1, "trichome_on_top", "morph_filter"],
        ["base1_128_grabcut",           False,  True,   True,   True, 128, 1, "trichome_on_top", "grabCut"],
    ]

    #parameters_to_change = [
    #    ["hp_trichomes", False, True, False, False, 224, 1, "trichome_on_top", "hp_filter" ],
    #    ["big_dots_only", False, True, False, False, 224, 1, "big_dots_only", "hp_filter" ],
    #    ["morph_trichomes", False, True, False, False, 224, 1, "trichome_on_top", "morph_filter" ],
    #    ["grabCut_trichomes", False, True, False, False, 224, 1, "trichome_on_top", "grabCut_filter" ],
    #]

    for selection in parameters_to_change:
        model_name = selection[0]
        do_test_train_split = selection[1]
        set_up_files = selection[2]
        create_model = selection[3]
        run_models = selection[4]
        patch_dim = selection[5]
        batch_size = selection[6]
        big_dots_type = selection[7]
        trichome_type = selection[8]

        print("***************************************************************", file=myoutput)
        print("***************************************************************", file=myoutput)
        print("The model {} has patch size {}x{}, batch size {}, dots {} trichome {}".format(model_name, patch_dim, patch_dim, batch_size, big_dots_type, trichome_type), file=myoutput)
        print("***************************************************************")
        print("***************************************************************")
        print("The model {} has patch size {}x{}, batch size {}, dots {} trichome {}".format(model_name, patch_dim, patch_dim, batch_size, big_dots_type, trichome_type))

        if set_up_files:

            if do_test_train_split:
                # Set up the directories
                patchRGB.makeFreshDir(dest_dir)
                # We should have test, train and models in that dir
                dirs = ["test", "train", "models"]
                create_temp_directories(dest_dir, dirs)
                train_dir = os.path.join(dest_dir, "train")
                test_dir = os.path.join(dest_dir, "test")
                model_dir =  os.path.join(dest_dir, "models")
                # Both the training and test dir need all these
                dirs = [s_unlab_im, s_unlab_pat, s_lab_im, s_lab_pat, s_lab_imr, s_dots_pat, s_dots_im, s_dots_imr, s_bwg_pat, s_bwg_imr, t_bwg_pat, t_bwg_im, visualise]
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
            else:
                train_dir = os.path.join(dest_dir, "train")
                test_dir = os.path.join(dest_dir, "test")
                model_dir =  os.path.join(dest_dir, "models")


            # Now prep what we can without a model:
            dirs = ["train", "test"]
            for d in dirs:
                mydir = os.path.join(dest_dir, d)
                print("Patch the labelled images")
                dirname_src = os.path.join(mydir, s_lab_im)
                dirname_dst = os.path.join(mydir, s_lab_pat)
                patchRGB.patchDir(dirname_src, dirname_dst, pheight, pwidth, 0, 0)

                print("Unpatch the labelled images (to account for crops)")
                dirname_src = os.path.join(mydir, s_lab_pat)
                dirname_dst = os.path.join(mydir, s_lab_imr)
                patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)

                print("Dots from ground truth (patches - gets the right image size)")
                dirname_src = os.path.join(mydir, s_lab_pat)
                dirname_dst = os.path.join(mydir, s_dots_pat)
                redDots.getDotMaskDir(dirname_src, dirname_dst)

                print("Unpatch the ground truth dot mask (to account for crops)")
                dirname_src = os.path.join(mydir, s_dots_pat)
                dirname_dst = os.path.join(mydir, s_dots_imr)
                patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)

                print("Enlarge the ground truth dots (for comparison)")
                dirname_src = os.path.join(mydir, s_lab_imr)
                dirname_dst = os.path.join(mydir, s_bwg_imr)
                #big_dots_only; trichome_on_top
                #hp_filter, morph_filter, grabCut,
                redDots.enlargeDotsDir(dirname_src, dirname_dst, dots_type=big_dots_type, trichome_type=trichome_type)

                print("Patch the ground truth big dots for network training")
                dirname_src = os.path.join(mydir, s_bwg_imr)
                dirname_dst = os.path.join(mydir, s_bwg_pat)
                patchRGB.patchDir(dirname_src, dirname_dst, pheight, pwidth, 0, 0)

                print("Dots from ground truth (images)")
                dirname_src = os.path.join(mydir, s_lab_im)
                dirname_dst = os.path.join(mydir, s_dots_im)
                redDots.getDotMaskDir(dirname_src, dirname_dst)

                print("Patch the source")
                dirname_src = os.path.join(mydir, s_unlab_im)
                dirname_dst = os.path.join(mydir, s_unlab_pat)
                patchRGB.patchDir(dirname_src, dirname_dst, pheight, pwidth, 0, 0)

                if evaluate_enlarge_dots:
                    print("Counting dots in dots images")
                    dirname_src = os.path.join(mydir, s_dots_imr)
                    print("Counting dots in processed dots images")
                    gtDots = redDots.countDotsDir2(dirname_src)
                    dirname_src = os.path.join(mydir, s_bwg_imr)
                    gtBigDots = redDots.countDotsDir2(dirname_src)
                    imageNames = redDots.createFileList(dirname_src)
                    gtDots_df = pd.DataFrame(gtDots, columns=['Ismaels', 'imagefile'])
                    gtBigDots_df = pd.DataFrame(gtBigDots, columns=['processed', 'imagefile'])

                    mydf = gtBigDots_df.merge(gtDots_df, on='imagefile', how='inner')
                    #mydf['model'] = os.path.basename(model_file)
                    mydf['wrong dots'] = abs(mydf['Ismaels'] - mydf['processed'])
                    mydf['model'] = model_name
                    mydf = mydf[['model', 'imagefile', 'Ismaels', 'processed', 'wrong dots']]
                    print(mydf.to_string())
                    print("Mean wrong dots in {} using {} = {}".format(d, model_name, mydf['wrong dots'].mean()))

                    # lastly, visualise (somehow...)
                    s_dots_im_files = redDots.createFileList(os.path.join(mydir, s_dots_imr))
                    s_bwg_imr_files = redDots.createFileList(os.path.join(mydir, s_bwg_imr))

                    s_dots_im_files_df = pd.DataFrame(s_dots_im_files, columns=['s_dots_im_files'])
                    s_dots_im_files_df['basename'] = s_dots_im_files_df["s_dots_im_files"].apply(lambda x: os.path.basename(x))
                    s_bwg_imr_files_df = pd.DataFrame(s_bwg_imr_files, columns=['s_bwg_imr_files'])
                    s_bwg_imr_files_df['basename'] = s_bwg_imr_files_df["s_bwg_imr_files"].apply(lambda x: os.path.basename(x))

                    mydf2 = s_dots_im_files_df.merge(s_bwg_imr_files_df, on='basename', how='inner')
                    #print(mydf2.to_string())
                    all_files = mydf2[['s_bwg_imr_files', 's_dots_im_files']]
                    all_files = [tuple(x) for x in all_files.to_numpy()]

                    #all_files = zip(t_bwg_im_files, s_dots_im_files)
                    dirname_dst = os.path.join(mydir, visualise)
                    for pair in all_files:
                        #print("Comparing {} with {}".format(pair[0], pair[1]))
                        img_mine = cv2.imread(pair[0])
                        img_gt = cv2.imread(pair[1])
                        #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)
                        img_mine[img_mine>250]=170
                        img_mine[img_gt>200]=255

                        imageBaseName = os.path.splitext(os.path.basename(pair[0]))[0]
                        save_name = os.path.join(dirname_dst, "{}.png".format(imageBaseName))
                        skimage.io.imsave(save_name, img_mine, check_contrast=False)

        else:
            train_dir = os.path.join(dest_dir, "train")
            test_dir = os.path.join(dest_dir, "test")
            model_dir =  os.path.join(dest_dir, "models")


        if create_model:
            print("Training a model!!!!!!!!!!!!!!!!!!!!!!!", file=myoutput)
            print("Training a model!!!!!!!!!!!!!!!!!!!!!!!")
            dirname_src = os.path.join(train_dir, s_unlab_pat)
            dirname_dst = os.path.join(train_dir, s_bwg_pat)
            src_images, tar_images = generatePictures.load_images2(dirname_src, dirname_dst)
            # Scale and convert to floats (because they're visible pixels right now)
            #print(src_images.shape)
            src_images = (src_images - 127.5) / 127.5
            tar_images = (tar_images - 127.5) / 127.5
            #print(src_images.shape)
            dataset = np.asarray([src_images, tar_images])
            image_shape = dataset[0].shape[1:]
            #print(image_shape)
            # define the models
            d_model = m.define_discriminator(image_shape)
            g_model = m.define_generator(image_shape)
            # define the composite model
            gan_model = m.define_gan(g_model, d_model, image_shape)
            # train model
            m.train(d_model, g_model, gan_model, dataset, n_epochs=50, n_batch=batch_size, destDir=model_dir, model_name=model_name)
            #m.train(d_model, g_model, gan_model, dataset, n_epochs=1, n_batch=batch_size, destDir=model_dir, model_name=model_name)

            model_files = redDots.createFileList(model_dir, formats=['.h5'])
        else:
           model_files = redDots.createFileList(model_dir, formats=['.h5'])
           #model_files = [model_file,]


        # Now run the models over the test files
        print(run_models)
        if run_models:
            print("Testing the models!!!!!!!!!!!!!!!!!!!!!!!", file=myoutput)
            print("Testing the models!!!!!!!!!!!!!!!!!!!!!!!")
            for model_file in model_files:
                print("Testing model {}".format(model_file))
                #for d in ["test", "train"]:
                for d in ["test",]:
                    mydir = os.path.join(dest_dir, d)
                    #print("Patch the source")
                    #dirname_src = os.path.join(mydir, s_unlab_im)
                    #dirname_dst = os.path.join(mydir, s_unlab_pat)
                    #patchRGB.patchDir(dirname_src, dirname_dst, pheight, pwidth, 0, 0)
                    # Style transfer the source patches
                    print("Translating the {} set".format(d))
                    dirname_src = os.path.join(mydir, s_unlab_pat)
                    dirname_dst = os.path.join(mydir, t_bwg_pat)
                    t.translate(model_file, dirname_src, dirname_dst)

                    # Tidy the style-transferred source patches
                    #dirname_src = os.path.join(dest_dir, t_bwg_pat)
                    #dirname_dst = os.path.join(dest_dir, t_bwg_pat)
                    #redDots.tidyImageDir(dirname_src, dirname_dst)


                    # Unpatch the translated image
                    print("unpatching the translated patches")
                    dirname_src = os.path.join(mydir, t_bwg_pat)
                    dirname_dst = os.path.join(mydir, t_bwg_im)
                    patchRGB.unpatchDir(dirname_src, dirname_dst, pheight, pwidth)

                    # Tidy the style-transferred source complete image
                    print("Tidying the image")
                    dirname_src = os.path.join(mydir, t_bwg_im)
                    dirname_dst = os.path.join(mydir, t_bwg_im)
                    redDots.tidyImageDir(dirname_src, dirname_dst, "")

                    # Count the dots
                    print("Counting dots in predicted images")
                    dirname_src = os.path.join(mydir, t_bwg_im)
                    foundDots = redDots.countDotsDir2(dirname_src)
                    print("Counting dots in dots images")
                    dirname_src = os.path.join(mydir, s_dots_imr)
                    print("Counting dots in processed dots images")
                    gtDots = redDots.countDotsDir2(dirname_src)
                    dirname_src = os.path.join(mydir, s_bwg_imr)
                    gtBigDots = redDots.countDotsDir2(dirname_src)
                    imageNames = redDots.createFileList(dirname_src)

                    #for i in zip(imageNames, foundDots, gtBigDots, gtDots):
                    #    basename = os.path.basename(model_file)
                    #    print("model {} file {} found cells {}; gt (processed): {}; gt (Ismael's) {}".format(basename, os.path.basename(i[0]), i[1], i[2], i[3]))

                    foundDots_df = pd.DataFrame(foundDots, columns=['foundDots', 'imagefile'])
                    gtDots_df = pd.DataFrame(gtDots, columns=['Ismaels', 'imagefile'])
                    gtBigDots_df = pd.DataFrame(gtBigDots, columns=['processed', 'imagefile'])

                    mydf = foundDots_df.merge(gtDots_df, on='imagefile', how='inner')
                    mydf = mydf.merge(gtBigDots_df, on='imagefile', how='inner')
                    mydf['model'] = os.path.basename(model_file)
                    mydf = mydf[['model', 'imagefile', 'foundDots', 'Ismaels', 'processed']]
                    mydf['pred-Ismaels'] = abs(mydf['foundDots'] - mydf['Ismaels'])
                    mydf['pred-proc'] = abs(mydf['foundDots'] - mydf['processed'])
                    print(mydf.to_string(), file=myoutput)
                    print("Mean wrong dots in {} using {} Ismaels {} vs processed {}".format(d, model_name, mydf['pred-Ismaels'].mean(), mydf['pred-proc'].mean()), file=myoutput)
                    print(mydf.to_string())
                    print("Mean wrong dots in {} using {} Ismaels {} vs processed {}".format(d, model_name, mydf['pred-Ismaels'].mean(), mydf['pred-proc'].mean()))
                    print("Mean dots in {} using {} Ismaels {} vs processed {}".format(d, model_name, mydf['Ismaels'].mean(), mydf['processed'].mean()))


                    # lastly, visualise (somehow...)
                    t_bwg_im_files = redDots.createFileList(os.path.join(mydir, t_bwg_im))
                    s_dots_im_files = redDots.createFileList(os.path.join(mydir, s_dots_imr))
                    s_bwg_imr_files = redDots.createFileList(os.path.join(mydir, s_bwg_imr))

                    t_bwg_im_files_df = pd.DataFrame(t_bwg_im_files, columns=['t_bwg_im_files'])
                    t_bwg_im_files_df['basename'] = t_bwg_im_files_df["t_bwg_im_files"].apply(lambda x: os.path.basename(x))
                    s_dots_im_files_df = pd.DataFrame(s_dots_im_files, columns=['s_dots_im_files'])
                    s_dots_im_files_df['basename'] = s_dots_im_files_df["s_dots_im_files"].apply(lambda x: os.path.basename(x))
                    s_bwg_imr_files_df = pd.DataFrame(s_bwg_imr_files, columns=['s_bwg_imr_files'])
                    s_bwg_imr_files_df['basename'] = s_bwg_imr_files_df["s_bwg_imr_files"].apply(lambda x: os.path.basename(x))

                    mydf2 = t_bwg_im_files_df.merge(s_dots_im_files_df, on='basename', how='inner')
                    mydf2 = mydf2.merge(s_bwg_imr_files_df, on='basename', how='inner')
                    #print(mydf2.to_string())
                    all_files = mydf2[['t_bwg_im_files', 's_dots_im_files']]
                    all_files = [tuple(x) for x in all_files.to_numpy()]

                    #all_files = zip(t_bwg_im_files, s_dots_im_files)
                    dirname_dst = os.path.join(mydir, visualise)
                    for pair in all_files:
                        #print("Comparing {} with {}".format(pair[0], pair[1]))
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
