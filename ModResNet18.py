######################################################
# Daniel Johnston
# LC19414
# CMSC 691
# Final Project
######################################################

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np



class ModResNetBlock(tf.keras.Model):
    def __init__(self,
                 num_filters,
                 kernel_size=3,
                 padding='same',
                 dropout=0.2,
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

class ConvSpread(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=3, padding='same', dropout=0.2, trainable=False):
        super().__init__()
        num_filters /= 4
        self.conv_1_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(2,2), trainable=trainable)
        self.relu_1_1 = layers.ReLU()
        self.bn_1_1 = layers.BatchNormalization()
        self.drop_1_1 = layers.Dropout(dropout)

        self.conv_1_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(4,4), trainable=trainable)
        self.relu_1_2 = layers.ReLU()
        self.bn_1_2 = layers.BatchNormalization()
        self.drop_1_2 = layers.Dropout(dropout)

        self.conv_1_4 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(8,8), trainable=trainable)
        self.relu_1_4 = layers.ReLU()
        self.bn_1_4 = layers.BatchNormalization()
        self.drop_1_4 = layers.Dropout(dropout)

        self.conv_1_8 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(16,16), trainable=trainable)
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



class ModResNet18BaseDownsample(tf.keras.Model):
    def __init__(self, trainable=False):
        super().__init__()
        self.conv_spread = ConvSpread(32, trainable=trainable)
        self.block_1 = ModResNetBlock(32, trainable=trainable, cat=True)
        self.block_2 = ModResNetBlock(64, trainable=trainable, cat=False)
        self.block_3 = ModResNetBlock(64, trainable=trainable, cat=True)
        self.block_4 = ModResNetBlock(128, trainable=trainable, cat=False)
        self.block_5 = ModResNetBlock(128, trainable=trainable, cat=True)
        self.block_6 = ModResNetBlock(256, trainable=trainable, cat=False)
        self.block_7 = ModResNetBlock(256, trainable=trainable, cat=True)
        self.block_8 = ModResNetBlock(512, trainable=trainable, cat=False)

        self.pool_1 = layers.MaxPool2D(strides=2)
        self.pool_2 = layers.MaxPool2D(strides=2)
        self.pool_3 = layers.MaxPool2D(strides=2)
        self.pool_4 = layers.MaxPool2D(strides=2)
        self.pool_5 = layers.MaxPool2D(strides=2)


    def call(self, inputs, trainable=False):
        # Add and cat
        x = self.conv_spread(inputs)
        x = self.pool_1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.pool_2(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.pool_3(x)
        x = self.block_6(x)
        x = self.pool_4(x)
        x = self.block_7(x)
        x = self.pool_5(x)
        x = self.block_8(x)

        return x


class ModResNet18BaseFullRes(tf.keras.Model):
    def __init__(self, trainable=False):
        super().__init__()
        self.conv_spread = ConvSpread(64, trainable=trainable)
        self.block_1 = ModResNetBlock(64, dropout=0.2, trainable=trainable, cat=False)
        self.block_2 = ModResNetBlock(64, dropout=0.2, trainable=trainable, cat=True)
        self.block_3 = ModResNetBlock(128, dropout=0.2, trainable=trainable, cat=False)
        self.block_4 = ModResNetBlock(128, dropout=0.2, trainable=trainable, cat=True)
        self.block_5 = ModResNetBlock(256, dropout=0.2, trainable=trainable, cat=False)
        self.block_6 = ModResNetBlock(256, dropout=0.2, trainable=trainable, cat=True)
        self.block_7 = ModResNetBlock(512, dropout=0.2, trainable=trainable, cat=False)
        self.block_8 = ModResNetBlock(512, dropout=0.2, trainable=trainable, cat=True)

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

class ModResNet18Classify(ModResNet18BaseDownsample):
    def __init__(self, batch_size=256, trainable=False):
        super().__init__(trainable=trainable)
        self.pool_g  = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(1000, input_shape=(batch_size,512), activation='softmax')

    def call(self, inputs, trainable=False):
        x = super().call(inputs, trainable)
        x = self.pool_g(x)
        x = self.dense_1(x)
        return x

    
class ModResNet18FcnFullRes(ModResNet18BaseFullRes):
    def __init__(self, classes=10, trainable=False):
        super().__init__(trainable=trainable)
        self.squeeze = layers.Conv2D(classes, 1, padding='valid', trainable=trainable)
        self.softmax = layers.Softmax()


    def call(self, inputs, trainable=False):
        x = super().call(inputs)
        x = self.squeeze(x)
        x = self.softmax(x)
        return x


class ModResNet18FcnEncodeDecode(ModResNet18BaseFullRes):
    def __init__(self, trainable=False):
        super().__init__(trainable=trainable)
        pass
        
    def call(self, inputs, trainable=False):
        pass
    

def normalize_and_resize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return tf.image.resize(image, [224,224]), label

def normalize(image, label):
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    model = ModResNet18FcnFullRes(trainable=True)
    model.build((1,224,224,3))
    model.summary()
    
    [ds_train, ds_test, ds_val], ds_info = tfds.load('imagenette', split=['train[:75%]', 'train[75%:]', 'validation'], shuffle_files=False, as_supervised=True, with_info=True)
    #print(ds_train)
    # ds_train = ds_train.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    one = ds_train.batch(1).take(1)
    for image, label in one:
        print(image.shape)
        out = model.predict(image)
        print(out.shape)
        print(out.argmax(axis=-1))

