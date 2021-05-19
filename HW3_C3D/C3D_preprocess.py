import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
from numpy import asarray

##################################################################
#        Convert UCF-101 dataset videos to input array           #
##################################################################

import skvideo.io

def convert_UCF(label_list, dir):
    testdir = dir
    testarray = []
    testlabel = []

    for Label in label_list:

        videosets_dir = testdir + Label + '/'
        for i in os.listdir(videosets_dir):

            videodata = skvideo.io.vread(videosets_dir + i)
            videolen = len(videodata)
            mask = np.zeros(videolen, dtype=np.bool)
            idx = [int(i / 16 * videolen) for i in range(16)]      # Select 16 frames evenly from the given video datasets
            mask[idx] = True

            videodata = videodata[mask].astype('int')
            print(i, idx, len(videodata))                          # Print video name / popped index / length

            temp = []
            for testimage in videodata:
                image = Image.fromarray(testimage.astype(np.uint8))
                image = image.crop((60, 20, 260, 220))
                img_resized = image.resize((96, 96), Image.BICUBIC)

                data = asarray(img_resized)
                temp.append(data)
            # print(np.asarray(temp, dtype = np.uint8).shape)
            testarray.append(temp)
            testlabel.append(Label)
    #print(np.asarray(testarray).shape)                             # Check the preprocessed data shape

    return testarray, testlabel

##################################################################
#    - Store datasets as .npy format                             #
#    - Display 10 example frames from the given dataset.         #
#    - The resulting shape will be (13320, 16, 96, 96, 3).       #
#    - Run this file in the same directory with the UCF dataset. #
##################################################################

def main():

    DIR = './UCF-101/'
    store_dir = './'
    LABEL_LIST = os.listdir(DIR)

    vid_array, vid_label = convert_UCF(LABEL_LIST, DIR)

    for i in range(10):
        sampleimage = Image.fromarray(np.asarray(vid_array[int(i*100)][i], dtype = np.uint8))
        plt.imshow(sampleimage)
        plt.show()

    vid_array = np.asarray( vid_array, dtype = np.uint8)
    np.save(store_dir+"ucf101_val.npy",  vid_array)
    np.save(store_dir+"ucf101_lab.npy", vid_label)

    ###### Below this is for testing whether labels are converted to np.array type ######

    Labels = np.load(store_dir + "ucf101_lab.npy")
    print(Labels[13200])

    mlb = MultiLabelBinarizer()
    labels = mlb.fit([Labels])
    # If we input label list into .fit(), then we get the incorrect output.
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
    print(mlb.classes_)
    Labels_bin = mlb.fit_transform(Labels)


if __name__ == '__main__':
    main()