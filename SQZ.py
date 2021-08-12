import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import regularizers


def load_para():
    classes = 3755
    batchsize = 160
    num_epoch = 5
    stepsperepoch = 400 * classes / batchsize
    size_input = 40
    DROPout = 0.5
    CONTACTS = 3
    weightier = 1e-4
    return classes, batchsize, num_epoch, stepsperepoch, size_input, DROPout, CONTACTS, weightier
num_classes, batch_size, epochs, steps_per_epoch, size, DROPOUT, CONCAT_AXIS, weight_decay = load_para()


def load_data():
    x_train_savepath = 'F:/PYCode/MOOK/label/ChineseCharacter/40/train_x.npy'
    y_train_savepath = 'F:/PYCode/MOOK/label/ChineseCharacter/40/train_y.npy'
    x_test_savepath = 'F:/PYCode/MOOK/label/ChineseCharacter/40/test_x.npy'
    y_test_savepath = 'F:/PYCode/MOOK/label/ChineseCharacter/40/test_y.npy'
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), size, size))
    x_test = np.reshape(x_test_save, (len(x_test_save), size, size))
    x_train = x_train.reshape(x_train.shape[0], size, size, 1)
    x_test = x_test.reshape(x_test.shape[0], size, size, 1)

    Y_train = np.zeros((1502000, num_classes), dtype='float32')
    Y_test = np.zeros((375500, num_classes), dtype='float32')
    for i in range(1502000):
        Y_train[i, y_train[i]] = 1
    for i in range(375500):
        Y_test[i, y_test[i]] = 1
    y_train = Y_train
    y_test = Y_test
    return x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test= load_data()


def fire_module(x, squeeze, expand, channel_axis, bypass_conv=0, bypass_simple=False, bypass_complex=False):
    sqz = Conv2D(filters=squeeze, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(weight_decay))(x)
    sqz = BatchNormalization(momentum=0.9, epsilon=1e-5)(sqz)
    sqz = Activation('relu')(sqz)

    expand_1 = Conv2D(filters=expand, kernel_size=(1, 1), strides=(1, 1), padding='same',
                  kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(weight_decay))(sqz)
    expand_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(expand_1)
    expand_1 = Activation('relu')(expand_1)

    expand_3 = Conv2D(filters=expand, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(weight_decay))(sqz)
    expand_3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(expand_3)
    expand_3 = Activation('relu')(expand_3)
    output = concatenate([expand_1, expand_3], axis=channel_axis)

    if bypass_simple:
        output = tf.math.add(output, x)
    if bypass_complex:
        x = Conv2D(bypass_conv, (1, 1), padding='same',
                   kernel_initializer="he_normal", kernel_regularizer=regularizers.l1(weight_decay))(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        output = tf.math.add(output, x)
    return output


def squeezenet():
    img_input = Input(shape=(size, size, 1))
    x = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(weight_decay))(img_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # fire 2,3,4
    x = fire_module(x, squeeze=16, expand=64, channel_axis=CONCAT_AXIS)
    x = fire_module(x, squeeze=16, expand=64, channel_axis=CONCAT_AXIS, bypass_simple=True)
    x = fire_module(x, squeeze=32, expand=128, channel_axis=CONCAT_AXIS)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # fire 5,6,7,8
    x = fire_module(x, squeeze=32, expand=128, channel_axis=CONCAT_AXIS, bypass_simple=True)
    x = fire_module(x, squeeze=48, expand=192, channel_axis=CONCAT_AXIS)
    x = fire_module(x, squeeze=48, expand=192, channel_axis=CONCAT_AXIS, bypass_simple=True)
    x = fire_module(x, squeeze=64, expand=256, channel_axis=CONCAT_AXIS)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # fire 9
    x = fire_module(x, squeeze=64, expand=256, channel_axis=CONCAT_AXIS, bypass_simple=True)
    x = Dropout(DROPOUT)(x)

    x = Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax')(x)
    sqz_model = Model(img_input, out)
    return sqz_model


model = squeezenet()
model.summary()

sgd1 = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
sgd2 = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9, nesterov=True)
sgd3 = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
sgd4 = tf.keras.optimizers.SGD(learning_rate=0.00005, momentum=0.9, nesterov=True)
sgd5 = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9, nesterov=True)


model.compile(loss='categorical_crossentropy',
              optimizer=sgd1,
              metrics=['categorical_accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=(x_test, y_test))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd2,
              metrics=['categorical_accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=(x_test, y_test))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd3,
              metrics=['categorical_accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=(x_test, y_test))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd4,
              metrics=['categorical_accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=(x_test, y_test))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd5,
              metrics=['categorical_accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=(x_test, y_test))

model.save('SqueezeNet.h5')

