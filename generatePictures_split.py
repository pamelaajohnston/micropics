#Largely taken from https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import os
import argparse

# load and scale the maps dataset ready for training
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


# load all images in a directory into memory - tar = target!
def load_images(path_src, path_tar, size=(256,256) ):
    src_list = list()
    tar_list = list()

    src_names = createFileList(path_src)
    tar_names = createFileList(path_tar)
    print(src_names)
    names = zip(src_names, tar_names)
    for src_filename, tar_filename in names:
        src_basename = os.path.splitext(os.path.basename(src_filename))[0]
        tar_basename = os.path.splitext(os.path.basename(tar_filename))[0]
        if (src_basename != tar_basename):
            print("Different files in the directories??!")
            quit()

        # load and resize the image
        src_pixels = load_img(src_filename, target_size=size)
        tar_pixels = load_img(tar_filename, target_size=size)
        # convert to numpy array
        src_img = img_to_array(src_pixels)
        tar_img = img_to_array(tar_pixels)
        # split into satellite and map
        src_list.append(src_img)
        tar_list.append(tar_img)

    return [asarray(src_list), asarray(tar_list)]

# load all images in a directory into memory - tar = target!
def load_images2(path_src, path_tar, size=(256,256) ):
    src_list1 = list()
    tar_list1 = list()
    src_list2 = list()
    tar_list2 = list()

    src_names = createFileList(path_src)
    half_way = len(src_names) // 2

    for i, src_filename in enumerate(src_names):
        src_basename = os.path.splitext(os.path.basename(src_filename))[0]
        tar_filename = os.path.join(path_tar, "{}.png".format(src_basename))
        # Check tar file exists
        if os.path.isfile(tar_filename):
            # load and resize the image
            src_pixels = load_img(src_filename, target_size=size)
            tar_pixels = load_img(tar_filename, target_size=size)
            # convert to numpy array
            src_img = img_to_array(src_pixels)
            tar_img = img_to_array(tar_pixels)
            # split into satellite and map
            if i < half_way:
                src_list1.append(src_img)
                tar_list2.append(tar_img)
            else:
                src_list2.append(src_img)
                tar_list1.append(tar_img)
        else:
            print("Source file for {} but no corresponding target file".format(src_basename))

    return [asarray(src_list1), asarray(tar_list1), asarray(src_list2), asarray(tar_list2)]

if __name__ == "__main__":
    # dataset path
    path_src = '../data/planktothrixCrop/redDots'
    path_tar = '../data/planktothrixCrop/labelledPics'
    filename1 = 'trichomes_256_1.npz'
    filename2 = 'trichomes_256_2.npz'

    parser = argparse.ArgumentParser(description="Takes a folder of images and creates 2 npz datasets")
    parser.add_argument("-s", "--source", help="the source directory")
    parser.add_argument("-t", "--target",   help="the directory with the target files")
    parser.add_argument("-o1", "--output1",   help="the first output file")
    parser.add_argument("-o2", "--output2",   help="the second output file")
    args = parser.parse_args()

    # We have "before" and "after" datasets with the same image in each.
    # This is for CGAN so we should have different "before" and "after" images.
    # So, we'll split them.

    if args.source:
        path_src = args.source
    if args.target:
        path_tar = args.target
    if args.output1:
        filename1 = args.output
    if args.output2:
        filename2 = args.output

    # load dataset
    [src_images1, tar_images1, src_images2, tar_images2] = load_images2(path_src, path_tar)
    print('Loaded: ', src_images1.shape, tar_images2.shape)
    # save as compressed numpy array
    savez_compressed(filename1, src_images1, tar_images1)
    savez_compressed(filename2, src_images2, tar_images2)
    print('Saved dataset: ', filename1)
    print('Saved dataset: ', filename2)

    quit()

    # load the prepared dataset
    from numpy import load
    from matplotlib import pyplot
    # load the dataset
    data = load(filename)
    src_images, tar_images = data['arr_0'], data['arr_1']
    print('Loaded: ', src_images.shape, tar_images.shape)
    # plot source images
    n_samples = 3
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(src_images[i].astype('uint8'))
    # plot target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(tar_images[i].astype('uint8'))
    pyplot.show()
