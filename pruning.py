import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from kerassurgeon.operations import delete_channels
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def load_para():
    classes = 3755
    batchsize = 160
    num_epoch = 1
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
    x_test_savepath = 'F:/PYCode/MOOC/label/ChineseCharacter/40/test_x.npy'
    y_test_savepath = 'F:/PYCode/MOOC/label/ChineseCharacter/40/test_y.npy'
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
x_train, y_train, x_test, y_test = load_data()
y_label = y_test



def cut_cnn_channel(model, num):
    cnn_layer = model.get_layer(index=num)
    weight_all = cnn_layer.get_weights()
    weight_filter = weight_all[0]
    num_filter = weight_filter.shape[-1]
    sum_filter = np.zeros([num_filter])
    for i in range(num_filter):
        sum_filter[i] = np.sum(np.abs(weight_filter[:, :, :, i]))
    num_small = np.argmin(sum_filter)
    cut_model = delete_channels(model, cnn_layer, [num_small], copy=True)
    return cut_model



sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model = keras.models.load_model('SqueezeNet.h5')
model.summary()




# ---------Fire剪枝---------#
# 5,8,9,    15,18,19,   25,28,29,   36,39,40    //具体层号
# 3,12,12   3,12,12     6,24,24     6,24,24     //剪枝数量（20%）

# 46,49,50, 56,59,60,   66,69,70,   77,80,81
# 9,36,36   9,36,36     12,48,48    12,48,48
def cut_fir(model, index, num_to_cut):
    print('going to cut', num_to_cut)
    out_model = cut_cnn_channel(model, index)
    for i in range(num_to_cut-1):
        out_model = cut_cnn_channel(out_model, index)
    return out_model

new_model = cut_fir(model, 5, 3)
new_model = cut_fir(new_model, 8, 12)
new_model = cut_fir(new_model, 9, 12)



sgd1 = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
sgd2 = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9, nesterov=True)
sgd3 = tf.keras.optimizers.SGD(learning_rate=0.000005, momentum=0.9, nesterov=True)


new_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd1,
                  metrics=['categorical_accuracy'])
new_model.fit(x_train, y_train, batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=3,
              validation_data=(x_test, y_test),
              validation_freq=3,
              shuffle=True)

new_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd2,
                  metrics=['categorical_accuracy'])
new_model.fit(x_train, y_train, batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=2,
              validation_freq=2,
              validation_data=(x_test, y_test),
              shuffle=True)

new_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd3,
                  metrics=['categorical_accuracy'])
new_model.fit(x_train, y_train, batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=1,
              validation_data=(x_test, y_test),
              shuffle=True)


print('1:保存网络   2:放弃    3.训练')
p = int(input())
if p == 1:
    new_model.summary()
    print('保存此次剪枝模型')
    new_model.save('fire1_3_12.h5')
elif p==2:
    print('放弃剪枝')
elif p==3:
    print('执行恢复训练')
    time = int(input())
    while time == 0:
        print('设置学习率')
        lr = float(input())
        print('设置迭代次数')
        ep = int(input())
        sgd4 = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        print('开始训练')
        new_model.compile(loss='categorical_crossentropy',
                          optimizer=sgd4,
                          metrics=['categorical_accuracy'])
        new_model.fit(x_train, y_train, batch_size=batch_size,
                      steps_per_epoch=steps_per_epoch,
                      epochs=ep,
                      validation_data=(x_test, y_test),
                      shuffle=True)
        print('0:继续训练   1:结束训练')
        time = int(input())
    print('3:保存网络   4:放弃')
    q = int(input())
    if q == 3:
        new_model.summary()
        print('保存此次剪枝模型')
        new_model.save('fire1_3_12.h5')
    else:
        print('放弃剪枝')



# ---------Conv剪枝---------#
'''
def cut_conv(model, index, to_cut):
    print('going to cut', to_cut)
    out_model = cut_cnn_channel(model, index)
    for i in range(to_cut - 1):
        out_model = cut_cnn_channel(out_model, index)
    return out_model

new_model = cut_conv(model, 1, 19)


sgd1 = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
sgd2 = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9, nesterov=True)
sgd3 = tf.keras.optimizers.SGD(learning_rate=0.000005, momentum=0.9, nesterov=True)


new_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd1,
                  metrics=['categorical_accuracy'])
new_model.fit(x_train, y_train, batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=3,
              validation_data=(x_test, y_test),
              shuffle=True)

new_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd2,
                  metrics=['categorical_accuracy'])
new_model.fit(x_train, y_train, batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=2,
              validation_data=(x_test, y_test), 
              shuffle=True)

new_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd3,
                  metrics=['categorical_accuracy'])
new_model.fit(x_train, y_train, batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              epochs=1,
              validation_data=(x_test, y_test), 
              shuffle=True)



print('1:保存网络   2:放弃    3.训练')
p = int(input())
if p == 1:
    new_model.summary()
    print('保存此次剪枝模型')
    new_model.save('conv1_19.h5')
elif p==2:
    print('放弃剪枝')
elif p == 3:
    print('执行恢复训练')
    time = int(input())
    while time == 0:
        print('设置学习率')
        lr = float(input())
        sgd4 = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        print('迭代次数')
        epoch = int(input())
        print('开始训练')
        new_model.compile(loss='categorical_crossentropy',
                          optimizer=sgd4,
                          metrics=['categorical_accuracy'])
        new_model.fit(x_train, y_train, batch_size=batch_size,
                      steps_per_epoch=steps_per_epoch,
                      epochs=epoch,
                      validation_data=(x_test, y_test), 
                      shuffle=True)
        print('0:继续训练   1:结束训练')
        time = int(input())
    print('3:保存网络   4:放弃')
    q = int(input())
    if q == 3:
        new_model.summary()
        print('保存此次剪枝模型')
        new_model.save('conv1_19.h5')
    else:
        print('放弃剪枝')
'''
