import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from tensorflow.keras.layers import Convolution3D, Dense, Dropout, Flatten, Activation, MaxPooling3D, ZeroPadding3D, Conv3D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.layers as layers

class ThreeDimCNN(object):

    def __init__(self, input_dim, input_frames, channel, batch_size, num_epochs, num_classes, data_dir):
        self.input_dim = input_dim
        self.input_frames = input_frames
        self.channel = channel
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.data_dir = data_dir
    
    def label_list(self):

        label_list = []
        for i in os.listdir(self.data_dir):
            label_list.append(i)
        print(label_list)

        return label_list

    def load_data(self, color=False, skip=True):

        store_dir = './'
        Y = []
        X = np.load(store_dir + "ucf101_val.npy")
        category = np.load(store_dir + "ucf101_lab.npy")
        label = self.label_list()
        for cat in category:
            temp = np.zeros(self.num_classes, dtype = np.uint8)
            temp[label.index(cat)] = 1
            Y.append(temp)
        return X, Y

    def model(self, input_shape):

        model = Sequential()

        wt_decay = 0.005
        (N, H, W, C) =input_shape
        # Conv - Pool layer set
        model.add(layers.BatchNormalization(input_shape=(N, H, W, C)))
        model.add(Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                         kernel_regularizer=l2(wt_decay), name="conv1a"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name="pool1"))

        model.add(Conv3D(128, (3,3,3), strides=(1,1,1), padding="same",
                         kernel_regularizer=l2(wt_decay), name="conv2a"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid", name="pool2"))

        model.add(Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding="same", activation="relu",
                         kernel_regularizer=l2(wt_decay), name="conv3a"))
        model.add(Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding="same",
                         kernel_regularizer=l2(wt_decay), name="conv3b"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool3"))

        model.add(layers.BatchNormalization())
        model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding="same", activation="relu",
                         kernel_regularizer=l2(wt_decay), name="conv4a"))
        model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding="same",
                         kernel_regularizer=l2(wt_decay), name="conv4b"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool4"))

        model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding="same", activation="relu",
                         kernel_regularizer=l2(wt_decay), name="conv5a"))
        model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding="same",
                         kernel_regularizer=l2(wt_decay), name="conv5b"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(ZeroPadding3D(padding=(0,1,1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool5"))
        # Flatten and FC layers
        model.add(Flatten())
        model.add(Dense(4096, activation="relu", name="fc6"))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation="relu", name="fc7"))
        model.add(Dropout(0.5))
        model.add(Dense(101, activation="softmax", name="fc8"))
        
        return model
'''
def preprocess(inputs):
    inputs /= 255
    inputs -= 0.5
    inputs *=2
    return inputs

def train_batches(train_X, train_Y, batch_size):
    N_tot = len(train_X)
    idx = [i for i in range(N_tot)]
    random.shuffle(idx)
    for i in range(int(N_tot/batch_size)):
        head = i*batch_size
        tail = (i+1)*batch_size
        temp_x = train_X[idx[head]:idx[tail]]
        temp_x = temp_x.astype('float')
        x = preprocess(temp_x)
        y = train_Y[idx[head]:idx[tail]]
        yield x, y

def dev_batches(test_X, test_Y, batch_size):
    N_tot = len(test_X)
    idx = [i for i in range(N_tot)]
    random.shuffle(idx)
    for i in range(int(N_tot / batch_size)):
        head = i * batch_size
        tail = (i + 1) * batch_size
        temp_x = test_X[idx[head]:idx[tail]]
        temp_x = temp_x.astype('float')
        x = preprocess(temp_x)
        y = test_Y[idx[head]:idx[tail]]
        yield x, y
'''

def main():

    data_dir = './UCF-101'

    three_dim = ThreeDimCNN(96, 16, 3, 32, 10, 101, data_dir)
    X_, Y_ = three_dim.load_data()
    (trainX, testX, trainY, testY) = train_test_split(X_, np.asarray(Y_), test_size=0.2, random_state=42)

    print(trainX.shape, testX.shape, trainY.shape)
    #lr = 5.e-5
    batchsize = 32
    epoch = 20
    #n_train, n_test = len(trainX), len(testX)

    model = three_dim.model(input_shape=(16, 96, 96, 3))
    lr_schedule = ExponentialDecay(initial_learning_rate=1.e-4,decay_steps=1000,decay_rate=0.5)
    adam = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1.e-8, decay=5.e-7, amsgrad=False)
    model.compile(optimizer=adam, loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    history = model.fit(trainX, trainY,batch_size=batchsize, epochs=epoch, verbose=1,
              validation_data=(testX, testY), validation_batch_size=batchsize)

    model.evaluate(testX, testY, batch_size=batchsize, verbose=1)

    # Plot Loss and Accuracy for Visualization

    output_save_path = './result_visualization/'
    if not os.path.exists('./result_visualization/'):
        os.mkdir('./result_visualization/')
    model.save_weights(output_save_path+'test_2.h5')

    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    with open(os.path.join(output_save_path, 'result.txt'), 'w') as file:
        file.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(epoch):
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, train_loss[i], train_acc[i], val_loss[i], val_acc[i]))
        file.close()

    fig, ax = plt.subplots(2, 1, figsize=(11, 17))
    ax[0].set_title("Overall loss", fontsize=15)
    ax[1].set_title("Overall accuracy", fontsize=15)
    ax[0].set_xlabel("Epoch", fontsize=13)
    ax[1].set_xlabel("Epoch", fontsize=13)
    ax[0].set_ylabel("Loss", fontsize=13)
    ax[1].set_ylabel("Accuracy", fontsize=13)
    ax[0].plot(history.history['loss'], label = "train")
    ax[0].plot(history.history['val_loss'], label="test")
    ax[1].plot(history.history['accuracy'], label = "train")
    ax[1].plot(history.history['val_accuracy'], label = "test")
    ax[0].legend()
    ax[1].legend()
    plt.show()

if __name__ == '__main__':
    main()

