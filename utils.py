###########################################
# Daniel Johnston
# LC19414
# CMSC 691
# Final Project
###########################################

import os
import cv2


# After attempting to load all of the images from the cityscapes
# dataset directly into memory for training, I realized that I
# don't have enough memory to do that.
# 
def create_path_files() :
    image_dir = "cityscapes_images/leftImg8bit"
    label_dir = "cityscapes_labels/gtFine"

    image_train_paths = []
    image_val_paths = []
    image_test_paths = []
    for subdir, dirs, files in os.walk(image_dir):
        for f in files:
            if (f[-3:] != 'png'):
                continue

            filepath = subdir + os.sep + f
            if 'train' in subdir:
                image_train_paths.append(filepath)
            elif 'val' in subdir:
                image_val_paths.append(filepath)
            else:
                image_test_paths.append(filepath)

    label_train_inst_paths = []
    label_train_full_paths = []
    label_val_inst_paths = []
    label_val_full_paths = []
    label_test_inst_paths = []
    label_test_full_paths = []
    for subdir, dirs, files in os.walk(label_dir):
        for f in files:
            if (f[-3:] != 'png'):
                continue

            filepath = subdir + os.sep + f
            if 'train' in subdir:
                if 'inst' in f:
                    label_train_inst_paths.append(filepath)
                elif 'label' in f:
                    label_train_full_paths.append(filepath)
            elif 'val' in subdir:
                if 'inst' in f:
                    label_val_inst_paths.append(filepath)
                elif 'label' in f:
                    label_val_full_paths.append(filepath)
            else:
                if 'inst' in f:
                    label_test_inst_paths.append(filepath)
                elif 'label' in f:
                    label_test_full_paths.append(filepath)

    image_train_paths.sort()
    image_val_paths.sort()
    image_test_paths.sort()
    with open('cityscapes_conf/image_train_paths.txt', 'w+') as f:
        for item in image_train_paths:
            f.write('{}\n'.format(item))
    with open('cityscapes_conf/image_val_paths.txt', 'w+') as f:
        for item in image_val_paths:
            f.write('{}\n'.format(item))
    with open('cityscapes_conf/image_test_paths.txt', 'w+') as f:
        for item in image_test_paths:
            f.write('{}\n'.format(item))

    label_train_inst_paths.sort()
    label_val_inst_paths.sort()
    label_test_inst_paths.sort()
    with open('cityscapes_conf/label_train_inst_paths.txt', 'w+') as f:
        for item in label_train_inst_paths:
            f.write('{}\n'.format(item))
    with open('cityscapes_conf/label_val_inst_paths.txt', 'w+') as f:
        for item in label_val_inst_paths:
            f.write('{}\n'.format(item))
    with open('cityscapes_conf/label_test_inst_paths.txt', 'w+') as f:
        for item in label_test_inst_paths:
            f.write('{}\n'.format(item))

    label_train_full_paths.sort()
    label_val_full_paths.sort()
    label_test_full_paths.sort()
    with open('cityscapes_conf/label_train_full_paths.txt', 'w+') as f:
        for item in label_train_full_paths:
            f.write('{}\n'.format(item))
    with open('cityscapes_conf/label_val_full_paths.txt', 'w+') as f:
        for item in label_val_full_paths:
            f.write('{}\n'.format(item))
    with open('cityscapes_conf/label_test_full_paths.txt', 'w+') as f:
        for item in label_test_full_paths:
            f.write('{}\n'.format(item))


if __name__ == '__main__':
    #create_path_files()
