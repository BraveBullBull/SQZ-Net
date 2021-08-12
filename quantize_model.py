import os
import logging
import pathlib
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers, Model
import tensorflow_model_optimization as tfmot
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_para():
    classes = 3755
    batchsize = 160
    num_epoch = 10
    stepsperepoch = 400 * classes / batchsize
    size_input = 40
    DROPout = 0.5
    CONTACTS = 3
    weightier = 1e-4
    return classes, batchsize, num_epoch, stepsperepoch, size_input, DROPout, CONTACTS, weightier
num_classes, batch_size, epochs, steps_per_epoch, size, DROPOUT, CONCAT_AXIS, weight_decay = load_para()



def load_data():
    x_train_savepath = 'F:/PYCode/MOOC/label/ChineseCharacter/40/train_x.npy'
    y_train_savepath = 'F:/PYCode/MOOC/label/ChineseCharacter/40/train_y.npy'
    x_test_savepath  = 'F:/PYCode/MOOC/label/ChineseCharacter/40/test_x.npy'
    y_test_savepath  = 'F:/PYCode/MOOC/label/ChineseCharacter/40/test_y.npy'

    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), size, size))
    x_test = np.reshape(x_test_save, (len(x_test_save), size, size))
    x_train = x_train.reshape(x_train.shape[0], size, size, 1)  # 给数据增加一个维度，使数据和网络结构匹配
    x_test = x_test.reshape(x_test.shape[0], size, size, 1)

    Y_train = np.zeros((1502000, num_classes), dtype='float32')
    Y_test = np.zeros((375500, num_classes), dtype='float32')
    for i in range(1502000):
        Y_train[i, y_train[i]] = 1
    for i in range(375500):
        Y_test[i, y_test[i]] = 1
    return x_train, Y_train, x_test, Y_test, y_test
x_train, y_train, x_test, y_test, y_label = load_data()


model = keras.models.load_model('19-fire5_6_24.h5')



sgd1 = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
sgd2 = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
sgd3 = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9, nesterov=True)
sgd4 = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
sgd5 = tf.keras.optimizers.SGD(learning_rate=0.00005, momentum=0.9, nesterov=True)
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

q_aware_model.compile(optimizer=sgd1,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
q_aware_model.fit(x_train, y_train, batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch,
                  epochs=3,
                  validation_freq=3,
                  validation_data=(x_test, y_test))


q_aware_model.compile(optimizer=sgd2,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
q_aware_model.fit(x_train, y_train, batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch,
                  epochs=3,
                  validation_freq=3,
                  validation_data=(x_test, y_test))

q_aware_model.compile(optimizer=sgd3,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
q_aware_model.fit(x_train, y_train, batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch,
                  epochs=4,
                  validation_freq=2,
                  validation_data=(x_test, y_test))

q_aware_model.compile(optimizer=sgd4,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
q_aware_model.fit(x_train, y_train, batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch,
                  epochs=4,
                  validation_freq=2,
                  validation_data=(x_test, y_test))

q_aware_model.compile(optimizer=sgd5,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
q_aware_model.fit(x_train, y_train, batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch,
                  epochs=5,
                  validation_data=(x_test, y_test))


print('1:保存网络   2:放弃    3.训练')
p = int(input())
if p == 1:
    q_aware_model.summary()
    print('保存模型')
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    tflite_name = "1111"
    open(tflite_name, "wb").write(quantized_tflite_model)
elif p==2:
    print('放弃量化')
elif p==3:
    print('执行量化训练')
    time = int(input())
    while time == 0:
        print('设置学习率')
        lr = float(input())
        print('迭代次数')
        epoch = int(input())
        sgd6 = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        print('开始训练')
        q_aware_model.compile(loss='categorical_crossentropy',
                              optimizer=sgd6,
                              metrics=['categorical_accuracy'])
        q_aware_model.fit(x_train, y_train, batch_size=batch_size,
                          steps_per_epoch=steps_per_epoch,
                          epochs=epoch)
        print('0:继续训练   1:结束训练')
        time = int(input())
    print('3:保存网络   4:放弃')
    q = int(input())
    if q == 3:
        q_aware_model.summary()
        print('保存模型')
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()
        tflite_name = "1111"
        open(tflite_name, "wb").write(quantized_tflite_model)
    else:
        print('放弃量化')








