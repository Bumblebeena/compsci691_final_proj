######################################################
# Daniel Johnston
# LC19414
# CMSC 691
# Final Project
######################################################
import os
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin')
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/lib64')
os.add_dll_directory(r'C:/Program Files/NVIDIA/CUDNN/v8.3/bin')
os.add_dll_directory(r'C:/Program Files/zlib/dll_x64')

import csv
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import VOC2012 as vc
from Utils import one_hot, TimeHistory


'''
Basic PARCnet block. Has two consecutive layers of 
'''
class PARCnetBlock(tf.keras.Model):
    def __init__(self,
                 num_filters,
                 kernel_size=3,
                 padding='same',
                 dropout=0.0,
                 trainable=False,
                 cat=True):
        super().__init__()
        num_filters /= 4
        dropout /= 4
        
        self.conv_1_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(1,1), trainable=trainable)
        self.relu_1_1 = layers.ReLU()
        self.bn_1_1 = layers.BatchNormalization()
        self.drop_1_1 = layers.Dropout(dropout)

        self.conv_1_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(2,2), trainable=trainable)
        self.relu_1_2 = layers.ReLU()
        self.bn_1_2 = layers.BatchNormalization()
        self.drop_1_2 = layers.Dropout(dropout)

        self.conv_1_4 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(4,4), trainable=trainable)
        self.relu_1_4 = layers.ReLU()
        self.bn_1_4 = layers.BatchNormalization()
        self.drop_1_4 = layers.Dropout(dropout)

        self.conv_1_8 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(8,8), trainable=trainable)
        self.relu_1_8 = layers.ReLU()
        self.bn_1_8 = layers.BatchNormalization()
        self.drop_1_8 = layers.Dropout(dropout)

        self.cat_1 = layers.Concatenate()
        
        self.conv_2_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(1,1), trainable=trainable)
        self.relu_2_1 = layers.ReLU()
        self.bn_2_1 = layers.BatchNormalization()
        self.drop_2_1 = layers.Dropout(dropout)
        
        self.conv_2_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(2,2), trainable=trainable)
        self.relu_2_2 = layers.ReLU()
        self.bn_2_2 = layers.BatchNormalization()
        self.drop_2_2 = layers.Dropout(dropout)

        self.conv_2_4 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(4,4), trainable=trainable)
        self.relu_2_4 = layers.ReLU()
        self.bn_2_4 = layers.BatchNormalization()
        self.drop_2_4 = layers.Dropout(dropout)

        self.conv_2_8 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(8,8), trainable=trainable)
        self.relu_2_8 = layers.ReLU()
        self.bn_2_8 = layers.BatchNormalization()
        self.drop_2_8 = layers.Dropout(dropout)
        
        self.cat_2 = layers.Concatenate()

        if (cat):
            self.cat_3 = layers.Concatenate()
        else:
            self.cat_3 = layers.Add()


    def call(self, inputs, trainable=False):
        x_1 = self.conv_1_1(inputs)
        x_2 = self.conv_1_2(inputs)
        x_4 = self.conv_1_4(inputs)
        x_8 = self.conv_1_8(inputs)
        x_1 = self.relu_1_1(x_1)
        x_2 = self.relu_1_2(x_2)
        x_4 = self.relu_1_4(x_4)
        x_8 = self.relu_1_8(x_8)
        x_1 = self.bn_1_1(x_1)
        x_2 = self.bn_1_2(x_2)
        x_4 = self.bn_1_4(x_4)
        x_8 = self.bn_1_8(x_8)
        if (trainable):
            x_1 = self.drop_1_1(x_1)
            x_2 = self.drop_1_2(x_2)
            x_4 = self.drop_1_4(x_4)
            x_8 = self.drop_1_8(x_8)
        x = self.cat_1([x_1, x_2, x_4, x_8])

        x_1 = self.conv_2_1(x)
        x_2 = self.conv_2_2(x)
        x_4 = self.conv_2_4(x)
        x_8 = self.conv_2_8(x)
        x_1 = self.relu_2_1(x_1)
        x_2 = self.relu_2_2(x_2)
        x_4 = self.relu_2_4(x_4)
        x_8 = self.relu_2_8(x_8)
        x_1 = self.bn_2_1(x_1)
        x_2 = self.bn_2_2(x_2)
        x_4 = self.bn_2_4(x_4)
        x_8 = self.bn_2_8(x_8)
        if (trainable):
            x_1 = self.drop_2_1(x_1)
            x_2 = self.drop_2_2(x_2)
            x_4 = self.drop_2_4(x_4)
            x_8 = self.drop_2_8(x_8)
        x = self.cat_2([x_1, x_2, x_4, x_8])

        x = self.cat_3([inputs, x])

        return x


'''
Like a PARCnet block except only one parallel layer and no skip connection
'''
class ConvSpread(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=3, padding='same', dropout=0.0, trainable=False):
        super().__init__()
        num_filters /= 4
        self.conv_1_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(1,1), trainable=trainable)
        self.relu_1_1 = layers.ReLU()
        self.bn_1_1 = layers.BatchNormalization()
        self.drop_1_1 = layers.Dropout(dropout)

        self.conv_1_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(2,2), trainable=trainable)
        self.relu_1_2 = layers.ReLU()
        self.bn_1_2 = layers.BatchNormalization()
        self.drop_1_2 = layers.Dropout(dropout)

        self.conv_1_4 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(4,4), trainable=trainable)
        self.relu_1_4 = layers.ReLU()
        self.bn_1_4 = layers.BatchNormalization()
        self.drop_1_4 = layers.Dropout(dropout)

        self.conv_1_8 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(8,8), trainable=trainable)
        self.relu_1_8 = layers.ReLU()
        self.bn_1_8 = layers.BatchNormalization()
        self.drop_1_8 = layers.Dropout(dropout)

        self.cat_1 = layers.Concatenate()

    def call(self, inputs, trainable=False):
        x_1 = self.conv_1_1(inputs)
        x_2 = self.conv_1_2(inputs)
        x_4 = self.conv_1_4(inputs)
        x_8 = self.conv_1_8(inputs)
        x_1 = self.relu_1_1(x_1)
        x_2 = self.relu_1_2(x_2)
        x_4 = self.relu_1_4(x_4)
        x_8 = self.relu_1_8(x_8)
        x_1 = self.bn_1_1(x_1)
        x_2 = self.bn_1_2(x_2)
        x_4 = self.bn_1_4(x_4)
        x_8 = self.bn_1_8(x_8)
        if (trainable):
            x_1 = self.drop_1_1(x_1)
            x_2 = self.drop_1_2(x_2)
            x_4 = self.drop_1_4(x_4)
            x_8 = self.drop_1_8(x_8)
        x = self.cat_1([x_1, x_2, x_4, x_8])

        return x


'''
Just the encoding side of a full PARCnet, with no output activation function.
Meant to be plugged into either the upsampling or classification parent class.
'''
class PARCnetEncode(tf.keras.Model):
    def __init__(self, trainable=False):
        super().__init__()
        self.conv_spread = ConvSpread(32, trainable=trainable)
        self.block_1 = PARCnetBlock(32, trainable=trainable, cat=True)
        self.block_2 = PARCnetBlock(64, trainable=trainable, cat=True)
        self.block_3 = PARCnetBlock(128, trainable=trainable, cat=True)
        self.block_4 = PARCnetBlock(256, trainable=trainable, cat=True)

        self.pool_1 = layers.MaxPool2D(strides=2)
        self.pool_2 = layers.MaxPool2D(strides=2)
        self.pool_3 = layers.MaxPool2D(strides=2)
        self.pool_4 = layers.MaxPool2D(strides=2)


    def call(self, inputs, trainable=False):
        x = self.conv_spread(inputs)
        x = self.pool_1(x)
        x = self.block_1(x)
        x = self.pool_2(x)
        x = self.block_2(x)
        x = self.pool_3(x)
        x = self.block_3(x)
        x = self.pool_4(x)
        x = self.block_4(x)

        return x


'''
The decoding side of a full PARCnet. Contains the output layer.
'''
class PARCnetSeg(PARCnetEncode):
    def __init__(self, classes, trainable=False):
        super().__init__(trainable)
        self.block_5 = ConvSpread(128, trainable=trainable)
        self.block_6 = ConvSpread(64, trainable=trainable)
        self.block_7 = ConvSpread(32, trainable=trainable)
        self.block_8 = ConvSpread(32, trainable=trainable)
        
        self.up_1 = layers.Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), padding='same')
        self.up_2 = layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='same')
        self.up_3 = layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='same')
        self.up_4 = layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='same')
        
        self.cat_1 = layers.Concatenate()
        self.cat_2 = layers.Concatenate()
        self.cat_3 = layers.Concatenate()
        self.cat_4 = layers.Concatenate()

        self.squeeze = layers.Conv2D(classes, 1, padding='valid', trainable=trainable)
        self.softmax = layers.Softmax()

        
    def call(self, inputs, trainable=False):
        a = self.conv_spread(inputs)
        x = self.pool_1(a)
        b = self.block_1(x)
        x = self.pool_2(b)
        c = self.block_2(x)
        x = self.pool_3(c)
        d = self.block_3(x)
        x = self.pool_4(d)
        x = self.block_4(x)

        # TODO: remove blocks?
        x = self.up_1(x)
        x = self.cat_1([x,d])
        x = self.block_5(x)
        x = self.up_2(x)
        x = self.cat_2([x,c])
        x = self.block_6(x)
        x = self.up_3(x)
        x = self.cat_3([x,b])
        x = self.block_7(x)
        x = self.up_4(x)
        x = self.cat_4([x,a])
        x = self.block_8(x)

        x = self.squeeze(x)
        x = self.softmax(x)

        return x


# Use this to pre train the net on imagenet data if initial direct training is bad.
# num_feature_maps should be set to however many feature maps are output by the
# last layer of PARCnetEncode
class PARCnetClassify(PARCnetEncode):
    def __init__(self, num_classes, batch_size=256, num_feature_maps=512, trainable=False):
        super().__init__(trainable)
        self.pool_g  = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(1000, input_shape=(batch_size,num_feature_maps), activation='softmax')

    def call(self, inputs, trainable=False):
        x = super().call(inputs, trainable)
        x = self.pool_g(x)
        x = self.dense_1(x)
        return x


# This class takes too much memory to run on my computer
class PARCnetFullResBase(tf.keras.Model):
    def __init__(self, trainable=False):
        super().__init__()
        self.conv_spread = ConvSpread(64, trainable=trainable)
        self.block_1 = PARCnetBlock(64, dropout=0.2, trainable=trainable, cat=False)
        self.block_2 = PARCnetBlock(64, dropout=0.2, trainable=trainable, cat=True)
        self.block_3 = PARCnetBlock(128, dropout=0.2, trainable=trainable, cat=False)
        self.block_4 = PARCnetBlock(128, dropout=0.2, trainable=trainable, cat=True)
        self.block_5 = PARCnetBlock(256, dropout=0.2, trainable=trainable, cat=False)
        self.block_6 = PARCnetBlock(256, dropout=0.2, trainable=trainable, cat=True)
        self.block_7 = PARCnetBlock(512, dropout=0.2, trainable=trainable, cat=False)
        self.block_8 = PARCnetBlock(512, dropout=0.2, trainable=trainable, cat=True)

    def call(self, inputs, trainable=False):
        # No pooling method
        x = self.conv_spread(inputs)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)

        return x

    
class PARCnetFullRes(PARCnetFullResBase):
    def __init__(self, classes=21, trainable=False):
        super().__init__(trainable=trainable)
        self.squeeze = layers.Conv2D(classes, 1, padding='valid', trainable=trainable)
        self.softmax = layers.Softmax()


    def call(self, inputs, trainable=False):
        x = super().call(inputs)
        x = self.squeeze(x)
        x = self.softmax(x)
        return x

    


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    batch_size = 10
    model = PARCnetEncoderDecoder(11, trainable=True)
    # model = PARCnetFullRes(trainable=True)
    # model.build((batch_size,224,224,3))
    # model.summary()
    
    # [ds_train, ds_test, ds_val], ds_info = tfds.load('imagenette', split=['train[:5%]', 'train[5%:7%]', 'validation[:2%]'], shuffle_files=False, as_supervised=True, with_info=True)
    # [ds_train, ds_test, ds_val], ds_info = tfds.load('imagenette', split=['train[:75%]', 'train[75%:]', 'validation'], shuffle_files=False, as_supervised=True, with_info=True)

    

    voc_base_dir = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    voc = vc.VOC2012(voc_base_dir, resize_method='resize', checkpaths=True)

    voc.load_train_data('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/voc2012_train')
    voc.load_test_data('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/voc2012_test')
    voc.load_val_data('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/voc2012_val')

    voc.convert_to_numpy()

    num_train = voc.train_images.shape[0]
    print('Num training images: {}'.format(num_train))
    num_test = voc.test_images.shape[0]
    print('Num test images: {}'.format(num_test))
    num_val = voc.val_images.shape[0]
    print('Num validation images: {}'.format(num_val))

    # # voc.train_images = tf.ragged.constant(voc.train_images)
    # # voc.train_labels = tf.ragged.constant(voc.train_labels)
    # # voc.test_images = tf.ragged.constant(voc.test_images)
    # # voc.test_labels = tf.ragged.constant(voc.test_labels)
    # # voc.val_images = tf.ragged.constant(voc.val_images)
    # # voc.val_labels = tf.ragged.constant(voc.val_labels)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # a, b = voc.get_batch_train(1)

    # print(a.shape)
    # out = model.predict(a)
    # print(out.shape)
    # print(out.argmax(axis=-1))

    # #####################################################
    # # https://www.tensorflow.org/tutorials/keras/save_and_load
    checkpoint_path = "./training{}".format(timestamp)
    checkpoint_path += "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        save_freq='epoch')
    # #####################################################

    # #####################################################
    # # from https://stackoverflow.com/questions/38445982/how-to-log-keras-loss-output-to-a-file
    csv_logger = CSVLogger('train_log_{}.csv'.format(timestamp), append=True, separator=',')
    # #####################################################

    time_callback = TimeHistory()
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy',tf.keras.metrics.OneHotMeanIoU(num_classes=11)])

    model.save_weights(checkpoint_path.format(epoch=0))    

    model.fit(x=voc.train_images[:600],
              y=one_hot(voc.train_labels[:600]),
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              callbacks = [cp_callback,csv_logger,time_callback],
              #validation_data=(voc.val_images[:50],one_hot(voc.val_labels[:50])),
              shuffle=True
              #validation_steps=batch_size
              )

    # with open('./train_times_{}.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)

    #     header = ['epoch', 'computation_time_ms']
    #     for i, time in enumerate(time_callback.times):
    #         writer.writerow([i, time])
        
    # @tf.function
    # def train_step(x, y):
    #     with tf.GradientTape() as tape:
    #         logits = model(x, training=True)
    #         batch_loss = loss_fn(y, logits)

    #     gradients = tape.gradient(batch_loss, model.trainable_weights)

    #     optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    #     correct = tf.math.count_nonzero(tf.equal(tf.math.argmax(logits, 1), tf.math.argmax(y,1)))

    #     return batch_loss, correct

    # @tf.function
    # def test_step(x, y):
    #     test_logits = model(x, training=False)
    #     test_loss = loss_fn(y, test_logits)
    #     test_correct = tf.math.count_nonzero(tf.equal(tf.math.argmax(test_logits, 1), tf.math.argmax(y, 1)))
    #     return test_loss, test_correct
    
    # num_epochs = 10

    
    # num_batches = int(num_train / batch_size)

    # train_loss = []
    # val_loss = []
    # train_acc = []
    # val_acc = []
    # for epoch in range(num_epochs):
    #     epoch_loss = 0
    #     epoch_correct = 0
    #     #for x_i, y_i in ds_train:
    #     for _ in range(num_batches):
    #         x_i, y_i = voc.get_batch_train(batch_size)
    #         batch_loss, batch_correct = train_step(x_i, one_hot(y_i))
    #         epoch_correct += batch_correct.numpy()
    #         epoch_loss += batch_loss

    #     train_loss.append(epoch_loss/num_batches)
    #     train_acc.append(epoch_correct/num_train)

    #     val_loss_sum = 0
    #     val_correct = 0
    #     for x_valid, y_valid in ds_val:
    #         batch_val_loss, batch_val_correct = test_step(x_valid, y_valid)
    #         val_loss_sum += batch_val_loss
    #         val_correct += batch_val_correct.numpy()
            
    #     val_loss.append(val_loss_sum/num_batches)
    #     val_acc.append(val_correct/num_valid)
    #     print("Completed epoch {} of {}".format(epoch+1, num_epochs))


    # # # Run our test set through the trained model
    # # x_test, y_test = 
    # # test_acc = np.ones(num_epochs)
    # # _, test_correct = test_step(x_test, y_test)
    # # test_acc *= (test_correct/num_test)


    # # Generate plots according to project requirements and save them to the
    # # output folder, which is guaranteed at this point to exist
    # epochs = np.arange(num_epochs)

    # plt.plot(epochs, np.array(train_acc)*100, label="train")
    # plt.plot(epochs, np.array(val_acc)*100, label="valid")
    # #plt.plot(epochs, test_acc*100, label="test")
    # plt.title("PARCnet Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy's")
    # plt.legend()
    # plt.grid()
    # plt.savefig("output/accuracy.png")
    # plt.close()

    # plt.plot(epochs, np.array(train_loss), label="train")
    # plt.plot(epochs, np.array(val_loss), label="valid")
    # plt.title("PARCnet Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss's")
    # plt.legend()
    # plt.grid()
    # plt.savefig("output/loss.png")
    # plt.close()
