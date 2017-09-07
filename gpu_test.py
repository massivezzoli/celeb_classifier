import argparse
# keras import
import numpy as np
np.random.seed(42)
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.engine.topology import Layer
from keras.utils import np_utils
from keras import backend as K
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

parser = argparse.ArgumentParser(description='get param for NN training')
parser.add_argument('-e', type=int, action="store", dest='epochs', default=10)
parser.add_argument('-g', action="store", dest='gpu')
parser.add_argument('-o', nargs='+', dest='opt')
parser.add_argument('-ls', type=int, nargs='+', dest='l_size')
args = parser.parse_args()
print(args.epochs)
print(args.opt)
print(args.l_size)
print(args.gpu)

mean_image = np.loadtxt(mean_image_path)

input("Press Enter to continue.")

# Get raw data
all_images, all_labels, all_bottles = load_preprocessed_data(corpus_dir, bottle_dir, img_dims)
#all_images = np.ravel(all_images).reshape(all_images.shape[0], all_images.shape[1] * all_images.shape[2] * all_images.shape[3])

# Get Dataset object
ds = Dataset(all_images, all_labels, all_bottles, split=split, one_hot=True, rnd_seed=rnd_seed)
n_features = ds.X.shape[1]


input("Press Enter to continue.")
#'/gpu:2'
with tf.device(args.gpu):
    beta = K.variable(1.)
    # Make a Scheduler:
    class RegScheduler(Callback):
        def __init__(self, beta):
            self.beta = beta

        def on_epoch_end(self, epoch, logs={}):
            initial_v = 1.00
            drop = 0.75
            epochs_drop = 10.00
            #K.set_value(self.beta, K.get_value(self.beta) * epoch**0.95)
            K.set_value(self.beta, (initial_v * np.power(drop, np.floor((1+epoch)/epochs_drop))))
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

            reg1 = cos_distance(bott, incep_bot) * (1-self.beta)
            self.add_loss(reg1)
            return bott
        def compute_output_shape(self, input_shap):
            return input_shape

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

    x = Dense(10, name='fc_2')(bot1)
    prediction = Activation('softmax')(x)

    print('layers made!')
    input("Press Enter to continue.")

    model = Model(inputs=[inputs1, inputs2], outputs=[prediction])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # loss='categorical_crossentropy'
    model.fit([ds.train.images, ds.train.bottles], ds.train.labels,
              batch_size=40, epochs=args.epochs, verbose=1, callbacks=[RegScheduler(beta=beta)])

    score, accuracy = model.evaluate([ds.test.images, ds.test.bottles], ds.test.labels, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', accuracy)

    # TESTING
    input("Press Enter to continue.")
    bot_lay_size = bot1.get_shape().as_list()[1]
    n_test_imgs = ds.train.images.shape[0]
    n_classes = ds.Y.shape[1]

    # added laerning phase to avoid issues with batchnorm and Dropout
    bottle_tensor_func = K.function([model.layers[0].input, K.learning_phase()],
                                        [model.get_layer('fc_1_act').output])

    # get test set bottleneck tensor:
    bottle_tensor_test = bottle_tensor_func([ds.test.images, 0])[0]

    bottle_tensor_train = np.zeros(shape=(n_test_imgs,bot_lay_size))
    bottle_labels_train = np.zeros(shape=(n_test_imgs, n_classes))
    counter = 0

    bot_batch = 20
    #get train set bottleneck tensor:
    for batch_x, batch_y, _ in ds.train.next_batch(bot_batch):
        bot_test = bottle_tensor_func([batch_x, 0])[0]
        bottle_tensor_train[counter:counter+bot_batch] = bot_test
        bottle_labels_train[counter:counter+bot_batch] = batch_y
        counter += bot_batch

    print(bottle_labels_train.shape)
    print(bottle_tensor_train.shape)

    final_bottle = np.vstack((bottle_tensor_train, bottle_tensor_test))
    final_labels = np.vstack((bottle_labels_train, ds.test.labels))

    #input("Press Enter to continue.")
    #tsne_output2(final_bottle, final_labels, 15, 25000, filename='merge_tsne.png')

    #tsne_val2, tsne_lab2 = tsne_output(bottle_tensor_test, ds.test.labels, 10, 25000, filename='test_tsne.png')
    #tsne_val1, tsne_lab1 = tsne_output(bottle_tensor_train, bottle_labels_train, 10, 25000, filename='train_tsne.png')
    #np.savetxt('tsne_val', tsne_val, delimiter=',')
    #np.savetxt('tsne_lab', tsne_lab, delimiter=',')
