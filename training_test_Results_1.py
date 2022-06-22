from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image

import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from keras.models import Sequential, load_model
from keras_layer_normalization import LayerNormalization

class Config:
  WD = "/exp/home/tliu/video_mouse/"
  DATASET_PATH = WD+"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
  SINGLE_TEST_PATH = WD+"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test032"
  BATCH_SIZE = 4
  EPOCHS = 15
  MODEL_PATH = WD+"model.hdf5"


def get_training_set():
    """
    Returns
    -------
    list
        A list of training sequences of shape (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    clips = []
    clips2 = []
    for vid in range(1,50):
        dirin = "images_grey/"+str(vid)+"/"
        num_images = len(listdir(dirin))
        all_frames = []
        for c in range(num_images):
            fin = dirin + str(c) + ".tiff"
            img = Image.open(fin).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            all_frames.append(img)
        for nclips in range(25):
            clip = np.zeros(shape=(10, 256, 256, 1))
            rind = np.sort(np.random.choice(num_images,10,replace=False))
            #clip[:,:,:,0] = all_frames[rind]
            for nclip in range(10):
                clip[nclip,:,:,0] = all_frames[rind[nclip]]
            if nclips <= 19:
                clips.append(clip)
            else:
                clips2.append(clip)
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
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
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
            batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)
    seq.save(Config.MODEL_PATH)
    return seq, test_set

def evaluate():
    model, test = get_model(True)
   
    layer_output = model.get_layer('encode').output
    intermediate_model = keras.Model(inputs=model.input, outputs=layer_output)
    intermediate_prediction=intermediate_model.predict(test, batch_size=4)
    np.save('own5_15_encode', intermediate_prediction) 

evaluate()


