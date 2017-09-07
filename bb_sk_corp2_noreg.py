# keras import
import tensorflow as tf
import numpy as np
import os
np.random.seed(42)
from keras.callbacks import Callback, LambdaCallback
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.engine.topology import Layer
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
# big boy utils
from utils.configuration import *
from utils.load_data import *
from utils.dataset import *
from utils.preprocessing import *
from utils.model import *
from utils.reporting import *
from utils.visualization import *
from utils.pca_tsne import *
from utils.load_preprocessed import *
# sklearn
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
lab_enc = LabelEncoder()

input("Press Enter to continue.")

# Get raw data
all_images1, all_labels1, all_bottles1 = load_preprocessed_data(corpus_dir, bottle_dir, img_dims)

trained_dir_2 = "/mnt/data/corpi/gaussian_masked_clr_20_50/"
corpus_dir_2 = os.path.join(trained_dir_2, "corpus")
bottle_dir_2 = os.path.join(trained_dir_2, "bottleneck")
# Get Raw data 2
all_images2, all_labels2, all_bottles2 = load_preprocessed_data(corpus_dir_2,
                                                                bottle_dir_2, img_dims)

# Split Data into train_test
i_train1, i_test1, b_train1, b_test1, l_train1, l_test1 = tts(all_images1, all_bottles1,
                                                      all_labels1, test_size=0.30,
                                                      stratify=all_labels1, random_state=42)

i_train2, i_test2, b_train2, b_test2, l_train2, l_test2 = tts(all_images2, all_bottles2,
                                                              all_labels2, test_size=0.30,
                                                              stratify=all_labels2, random_state=42)
imgs_train = np.vstack((i_train1, i_train2))
bots_train = np.vstack((b_train1, b_train2))
lab_train = np.concatenate((l_train1, l_train2), axis=0)
#Get test images
imgs_test = np.vstack((i_test1, i_test2))
bots_test = np.vstack((b_test1, b_test2))
lab_test = np.concatenate((l_test1, l_test2), axis=0)

# Perform one-hot encoding on all labels
lab_train_le = lab_enc.fit_transform(lab_train)
lab_train_ohe = enc.fit_transform(lab_train_le.reshape(-1,1)).toarray()
lab_test_le = lab_enc.fit_transform(lab_test)
lab_test_ohe = enc.fit_transform(lab_test_le.reshape(-1,1)).toarray()

#Split train into train and validation
imgs_val, imgs_test, bots_val, bots_test, lab_val_ohe, lab_test_ohe = tts(imgs_test, bots_test,
                                                                  lab_test_ohe, test_size=0.50,
                                                                  stratify=lab_test_ohe, random_state=42)

print(imgs_train.shape)
print(imgs_val.shape)
print(imgs_test.shape)
print(lab_train_ohe.shape)
print(lab_val_ohe.shape)
print(lab_test_ohe.shape)

input("Press Enter to continue.")

# Make a Scheduler:
epoch_count = K.variable(0)
beta = K.variable(1.)
class RegScheduler(Callback):
    def __init__(self, beta, epoch_count):
        self.beta = beta
        self.epoch_count = epoch_count
    def on_epoch_begin(self, epoch, logs={}):
        K.set_value(self.epoch_count, epoch)
    def on_epoch_end(self, epoch, logs={}):
        max_epoch= 70
        power = 4
        stop = 0
        K.set_value(self.beta, ((1-(epoch/max_epoch)) ** power ) * (1-stop) + stop )
        print('---current beta: %.3f' % K.get_value(beta))

# checkpoint load
model_dir = "./mdl/"
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'model_noreg_150.h5')
if os.path.exists(model_path):
    print('Loading model...')
    #for name in glob.glob('./mdl/model?.txt'):
    model = load_model(model_path)
else:
    print('Building Model..')
    # Make the layers:
    inputs1 = Input(shape=(150, 150, 3))

    x = Convolution2D(32, (3, 3), padding='same', name='c0')(inputs1)
    x = BatchNormalization(name='c0_bn')(x)
    x = Activation('relu', name='c0_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='c0_max')(x)

    x = Convolution2D(32, (3, 3), padding='same', name='c1')(x)
    x = BatchNormalization(name='c1_bn')(x)
    x = Activation('relu', name='c1_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='c1_max')(x)

    x = Convolution2D(32, (3, 3), padding='same', name='c2')(x)
    x = BatchNormalization(name='c2_bn')(x)
    x = Activation('relu', name='c2_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='c2_max')(x)

    x = Flatten(name='flat_0')(x)

    x = Dense(2048, name='fc_0')(x)
    x = BatchNormalization(name='fc_0_bn')(x)
    x = Activation('sigmoid', name='fc_0_act')(x)
    x = Dropout(0.7, name='fc_0_drop')(x)

    x = Dense(2048, name='fc_1')(x)
    x = BatchNormalization(name='fc_1_bn')(x)
    x = Activation('sigmoid', name='fc_1_act')(x)

    x = Dense(20, name='fc_2')(x)
    prediction = Activation('softmax')(x)

    model = Model(inputs=[inputs1], outputs=[prediction])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

print('layers made!')
#model.summary()

input("Press Enter to continue.")
# Training:
try:
    model.fit([imgs_train], lab_train_ohe,
              batch_size=50, epochs=80, verbose=1,
              validation_data=([imgs_val], lab_val_ohe),
              callbacks=[RegScheduler(beta=beta, epoch_count=epoch_count)])

    score, accuracy = model.evaluate([imgs_test], lab_test_ohe, batch_size=100, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', accuracy)

    epc_count= int(K.get_value(epoch_count))+1
    file_name = 'model2_noreg_'+str(epc_count)+'.h5'
    model_path_save = os.path.join(model_dir, file_name)
    model.save(model_path_save)

    # Layers sizes
    input("Press Enter to continue.")
    bot_lay_size = 2048
    n_train_imgs = imgs_train.shape[0]
    n_test_imgs = imgs_test.shape[0]
    n_classes = lab_train_ohe.shape[1]
    print('train images', n_train_imgs)
    print('test images', n_test_imgs)

    # backend function to accesss values from bottle layer
    bottle_tensor_func = K.function([model.layers[0].input, K.learning_phase()],
                                        [model.get_layer('fc_1_act').output])
    #set up np.array to store values for all images
    bottle_tensor_train = np.zeros(shape=(n_train_imgs, bot_lay_size))
    bottle_labels_train = np.zeros(shape=(n_train_imgs, n_classes))
    bottle_tensor_test = np.zeros(shape=(n_test_imgs, bot_lay_size))
    bottle_labels_test = np.zeros(shape=(n_test_imgs, n_classes))

    def batcher(X_train, y_train, size):
        X_batch = [X_train[indx:indx + size] for indx in range(0, len(X_train), size)]
        y_batch = [y_train[indx:indx + size] for indx in range(0, len(y_train), size)]
        return zip(X_batch, y_batch)

    counter = 0
    bot_batch = 20
    # get train set bottleneck activation values:
    for batch_x, batch_y in batcher(imgs_train, lab_train_ohe, bot_batch):
        bot_train = bottle_tensor_func([batch_x, 0])[0]
        bottle_tensor_train[counter:counter+bot_batch] = bot_train
        bottle_labels_train[counter:counter+bot_batch] = batch_y
        counter += bot_batch

    # get test set bottleneck activation values:
    counter = 0
    bot_batch = 20
    for batch_x, batch_y in batcher(imgs_test, lab_test_ohe, bot_batch):
        bot_train = bottle_tensor_func([batch_x, 0])[0]
        bottle_tensor_test[counter:counter+bot_batch] = bot_train
        bottle_labels_test[counter:counter+bot_batch] = batch_y
        counter += bot_batch
    # stack values in signle np.array
    final_bottle = np.vstack((bottle_tensor_train, bottle_tensor_test))
    final_labels = np.vstack((bottle_labels_train, bottle_labels_test))

    tsne_output2(final_bottle, final_labels, 30, 5000, n_train_imgs, filename='merge_tsne.png')
    K.clear_session()

except KeyboardInterrupt:
    score, accuracy = model.evaluate([imgs_test1], lab_test_ohe, batch_size=50, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', accuracy)
    epc_count= int(K.get_value(epoch_count))+1
    file_name = 'model2_noreg_'+str(epc_count)+'.h5'
    model_path_save = os.path.join(model_dir, file_name)
    model.save(model_path_save)
    K.clear_session()
