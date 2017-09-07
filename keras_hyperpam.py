# keras import
from __future__ import print_function
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from keras.callbacks import Callback
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.engine.topology import Layer
from keras.utils import np_utils
from keras import backend
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

#face_cascade = cv2.CascadeClassifier(face_cascade_path)
#eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
mean_image = np.loadtxt(mean_image_path)

input("Press Enter to continue.")

def data():
    np.random.seed(42)
    all_images, all_labels, all_bottles = load_preprocessed_data(corpus_dir, bottle_dir, img_dims)
    ds = Dataset(all_images, all_labels, all_bottles, split=split, one_hot=True, rnd_seed=rnd_seed)
    x_train=ds.train.images
    z_train=ds.train.bottles
    y_train=ds.train.labels
    x_test=ds.test.images
    z_test=ds.test.bottles
    y_test=ds.test.labels
    return x_train, z_train, y_train, x_test, z_test, y_test

input("Press Enter to continue.")
def model(x_train, z_train, y_train, x_test, z_test, y_test):
    beta = backend.variable(1.)

    class RegScheduler(Callback):
        def __init__(self, beta):
            self.beta = beta
        def on_epoch_end(self, epoch, logs={}):
            initial_v = 1.00
            drop = 0.75
            epochs_drop = 10.00
            backend.set_value(self.beta, (initial_v * np.power(drop, np.floor((1+epoch)/epochs_drop))))
    class BottleReg(Layer):
        def __init__(self, beta, **kwargs):
            super(BottleReg, self).__init__(**kwargs)
            self.beta=beta
        def call(self, x, mask=None):
            incep_bot = x[0]
            bott = x[1]
            def cos_distance(a1, b1):
                a1 = backend.l2_normalize(a1, axis=-1)
                b1 = backend.l2_normalize(b1, axis=-1)
                return backend.mean(1 - backend.sum((a1 * b1), axis=-1))
            reg1 = cos_distance(bott, incep_bot) * (1-self.beta)
            self.add_loss(reg1)
            return bott
        def compute_output_shape(self, input_shap):
            return input_shape

    inputs1 = Input(shape=(150, 150, 3))
    inputs2 = Input(shape=(2048,))

    x = Convolution2D(32, (3, 3), padding='same', name='c0')(inputs1)
    x = BatchNormalization(name='c0_bn')(x)
    x = Activation({{choice(['relu', 'tanh'])}}, name='c0_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='c0_max')(x)

    x = Convolution2D(32, (3, 3), padding='same', name='c1')(x)
    x = BatchNormalization(name='c1_bn')(x)
    x = Activation({{choice(['relu', 'tanh'])}}, name='c1_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='c1_max')(x)

    x = Convolution2D(32, (3, 3), padding='same', name='c2')(x)
    x = BatchNormalization(name='c2_bn')(x)
    x = Activation({{choice(['relu', 'tanh'])}}, name='c2_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='c2_max')(x)

    x = Flatten(name='flat_0')(x)

    x = Dense(2048, name='fc_0')(x)
    x = BatchNormalization(name='fc_0_bn')(x)
    x = Activation('sigmoid', name='fc_0_act')(x)

    x = Dropout({{uniform(0.4, 1)}}, name='fc_0_drop')(x)

    x = Dense(2048, name='fc_1')(x)
    x = BatchNormalization(name='fc_1_bn')(x)
    x = Activation('sigmoid', name='fc_1_act')(x)
    bot1 = BottleReg(beta)([inputs2, x])

    x = Dense(10, name='fc_2')(bot1)
    prediction = Activation('softmax')(x)

    model = Model(inputs=[inputs1, inputs2], outputs=[prediction])

    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, metrics=['accuracy'])

    model.fit([x_train, z_train], y_train,
              batch_size={{choice([20, 40])}}, epochs=50, verbose=0, callbacks=[RegScheduler(beta=beta)])

    score, accuracy = model.evaluate([x_test, z_test], y_test, verbose=0)
    print('Test accuracy:', accuracy)
    return{'loss': -accuracy, 'status':STATUS_OK, 'model':model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=5,
                              trials=Trials())
    print(best_run)
    #print("Evalutation of best performing model:")
    #print(best_model.evaluate(ds.test.iamges, ds.test.bottles, ds.test.labels))
