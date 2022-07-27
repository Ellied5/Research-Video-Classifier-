#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import torchvision 
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2 
import random 
import torch 
import keras
import tensorflow as tf 
import time 
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
import keras.layers.normalization
from keras.layers import LayerNormalization
from tensorflow.keras.callbacks import TensorBoard


class Config:
  WD = "/exp/home/tliu/video_mouse/"
  DATASET_PATH = WD+"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
  SINGLE_TEST_PATH = WD+"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test032"
  BATCH_SIZE = 3
  EPOCHS = 15
  MODEL_PATH = WD+"model.hdf5"


def get_training_set():
    """
    Returns
    -------
    list
        A list of training sequences of shape (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """

    DataDir="F:/2022_Summer/data"
    Categories=["Post-Injury", "Pre-Injury"]
    trainingdata=[] 
    clips = []
    clips2 = []

#creates a data set of images and labels 
    print("Training data is loading...")
    for category in Categories: 
        path=os.path.join(DataDir,category)
        num_images = len(listdir(path))
        for vid in os.listdir(path):
            print(len(os.listdir(path)))
            vid_array=[]
            vpath=os.path.join(path,vid)
            j=0
            for frame in os.listdir(vpath): 
                print(len(os.listdir(vpath)))
                img_array=cv2.imread(os.path.join(vpath,frame),0)
                img_array.resize(256,256)
                if j==0: 
                    vid_array=img_array
                else:
                    vid_array=np.concatenate([vid_array,img_array])
                j=j+1
            vid_array=np.array(vid_array)
            vid_array=vid_array.reshape(10,256,256)
            trainingdata.append([vid_array])
    X=[]
    for features in trainingdata:
        X.append(features)
    X=np.asarray(X)
    print(X.shape)
    X=X.reshape(X.shape[0],10,256,256,1)
    random.shuffle(X)
    
    for n in range(X.shape[0]):
        clip = X[n]

        if n%5==0:
            clips2.append(clip)
        else:
            clips.append(clip)
    clips=np.array(clips)
    clips2=np.array(clips2)
    return clips, clips2 

    

def get_model(reload_model=True):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    if not reload_model:
        return load_model(Config.MODEL_PATH,custom_objects={'LayerNormalization': LayerNormalization})
    training_set, test_set = get_training_set()
    training_set, test_set = np.array(training_set), np.array(test_set)
    print('Training set shape: ', training_set.shape)
    print('Test set shape: ', test_set.shape)
    NAME="Pre-vs-Post-Injury-ConvLSTM-{}".format(int(time.time()))

    tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(NAME), histogram_freq=1, profile_batch = 0)
   
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None,10, 256, 256, 1)))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(8, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization()) 
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    seq.layers[8]._name = 'encode'
    print(seq.summary())
    seq.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
    seq.fit(training_set, training_set,
            batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False, callbacks=[tensorboard, tensorboard_callback])
    seq.save(Config.MODEL_PATH)
    return seq, test_set

def evaluate():
    model, test = get_model(True)

    layer_output = model.get_layer('encode').output
    intermediate_model = keras.Model(inputs=model.input, outputs=layer_output)
    intermediate_prediction=intermediate_model.predict(test, batch_size=4)
    np.save('own5_15_encode', intermediate_prediction) 

evaluate()


# In[3]:





# In[ ]:




