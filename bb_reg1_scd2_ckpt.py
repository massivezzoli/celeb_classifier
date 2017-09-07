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

input("Press Enter to continue.")

# Get raw data
all_images, all_labels, all_bottles = load_preprocessed_data(corpus_dir, bottle_dir, img_dims)

# Get Dataset object
ds = Dataset(all_images, all_labels, all_bottles, split=split, one_hot=True, rnd_seed=rnd_seed)
n_features = ds.X.shape[1]

input("Press Enter to continue.")

# Make a Scheduler:
beta = K.variable(1.)
class RegScheduler(Callback):
    def __init__(self, beta):
        self.beta = beta
    def on_epoch_end(self, epoch, logs={}):
        max_epoch= 150
        power = 4
        stop = 0
        K.set_value(self.beta, ((1-(epoch/max_epoch)) ** power ) * (1-stop) + stop )
        print('---current beta: %.3f' % K.get_value(beta))

# Make a Custom Layer:
class BottleReg(Layer):
    def __init__(self, beta, **kwargs):
        super(BottleReg, self).__init__(**kwargs)
        self.beta=beta

    def call(self, x, mask=None):
        incep_bot = x[0]
        bott = x[1]
        def cos_distance(a1, b1):
            a1 = K.l2_normalize(a1, axis=-1)
            b1 = K.l2_normalize(b1, axis=-1)
            return K.mean(1 - K.sum((a1 * b1), axis=-1))
        reg1 = cos_distance(bott, incep_bot)*(1-self.beta)
        self.add_loss(reg1)
        return bott
    def compute_output_shape(self, input_shape):
        return input_shape

epoch_count = LambdaCallback(on_epoch_begin=lambda epoch, logs: print(epoch))

# checkpoint load
model_dir = "./mdl/"
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'model.h5')
if os.path.exists(model_path):
    print('Loading model...')
    model = load_model(model_path, custom_objects={'BottlerReg': BottleReg})
else:
    print('Building Model..')
    # Make the layers:
    inputs1 = Input(shape=(150, 150, 3))
    inputs2 = Input(shape=(2048,))

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
    x = Dropout(0.6, name='fc_0_drop')(x)

    x = Dense(2048, name='fc_1')(x)
    x = BatchNormalization(name='fc_1_bn')(x)
    x = Activation('sigmoid', name='fc_1_act')(x)
    bot1 = BottleReg(beta)([inputs2, x])

    x = Dense(20, name='fc_2')(bot1)
    prediction = Activation('softmax')(x)

    model = Model(inputs=[inputs1, inputs2], outputs=[prediction])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

print('layers made!')
model.summary()

input("Press Enter to continue.")
# Training:
try:
    model.fit([ds.train.images, ds.train.bottles], ds.train.labels,
              batch_size=50, epochs=150, verbose=1,
              validation_data=([ds.valid.images, ds.valid.bottles], ds.valid.labels),
              callbacks=[RegScheduler(beta=beta)])

    score, accuracy = model.evaluate([ds.test.images, ds.test.bottles], ds.test.labels, batch_size=100, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', accuracy)
    file_name = 'model_'+str(epoch_count)+'.h5'
    model_path_save = os.path.join(model_dir, file_name)
    model.save(model_path_save)


    # Layers sizes
    input("Press Enter to continue.")
    bot_lay_size = bot1.get_shape().as_list()[1]
    n_train_imgs = ds.train.images.shape[0]
    n_test_imgs = ds.test.images.shape[0]
    n_classes = ds.Y.shape[1]

    # backend function to accesss values from bottle layer
    bottle_tensor_func = K.function([model.layers[0].input, K.learning_phase()],
                                        [model.get_layer('fc_1_act').output])
    #set up np.array to store values for all images
    bottle_tensor_train = np.zeros(shape=(n_train_imgs,bot_lay_size))
    bottle_labels_train = np.zeros(shape=(n_train_imgs, n_classes))
    bottle_tensor_test = np.zeros(shape=(n_test_imgs,bot_lay_size))
    bottle_labels_test = np.zeros(shape=(n_test_imgs, n_classes))

    counter = 0
    bot_batch = 20
    # get train set bottleneck activation values:
    for batch_x, batch_y, _ in ds.train.next_batch(bot_batch):
        bot_train = bottle_tensor_func([batch_x, 0])[0]
        bottle_tensor_train[counter:counter+bot_batch] = bot_train
        bottle_labels_train[counter:counter+bot_batch] = batch_y
        counter += bot_batch
    # get test set bottleneck activation values:
    counter = 0
    for batch_x, batch_y, _ in ds.test.next_batch(bot_batch):
        bot_test = bottle_tensor_func([batch_x, 0])[0]
        bottle_tensor_test[counter:counter+bot_batch] = bot_test
        bottle_labels_test[counter:counter+bot_batch] = batch_y
        counter += bot_batch
    # stack values in signle np.array
    final_bottle = np.vstack((bottle_tensor_train, bottle_tensor_test))
    final_labels = np.vstack((bottle_labels_train, bottle_labels_test))

    tsne_output2(final_bottle, final_labels, 30, 5000, filename='merge_tsne.png')
    K.clear_session()

except KeyboardInterrupt:
    score, accuracy = model.evaluate([ds.test.images, ds.test.bottles], ds.test.labels, batch_size=batch_size, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', accuracy)
    file_name = 'model_'+epoch_count+'.h5'
    model_path_save = os.path.join(model_dir, file_name)
    model.save(model_path_save)
    K.clear_session()
