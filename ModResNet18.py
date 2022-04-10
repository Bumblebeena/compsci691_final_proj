######################################################
# Daniel Johnston
# LC19414
# CMSC 691
# Final Project
######################################################

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np



class ModResNetBlock(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=3, padding='same'):
        super().__init__()
        num_filters /= 4
        self.conv_1_1 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(1,1))
        self.bn_1_1 = layers.BatchNormalization()
        self.relu_1_1 = layers.ReLU()
        self.conv_1_2 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(2,2))
        self.bn_1_2 = layers.BatchNormalization()
        self.relu_1_2 = layers.ReLU()
        self.conv_1_4 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(4,4))
        self.bn_1_4 = layers.BatchNormalization()
        self.relu_1_8 = layers.ReLU()
        self.conv_1_8 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(8,8))
        self.bn_1_8 = layers.BatchNormalization()
        self.relu_1_8 = layers.ReLU()
        self.cat_1 = layers.Concatenate()
        
        self.conv_2_1 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(1,1))
        self.bn_2_1 = layers.BatchNormalization()
        self.relu_2_1 = layers.ReLU()
        self.conv_2_2 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(2,2))
        self.bn_2_2 = layers.BatchNormalization()
        self.relu_2_2 = layers.ReLU()
        self.conv_2_4 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(4,4))
        self.bn_2_4 = layers.BatchNormalization()
        self.relu_2_8 = layers.ReLU()
        self.conv_2_8 = layers.Conv2D(num_filters, kernel_size, padding, dilation_rate=(8,8))
        self.bn_2_8 = layers.BatchNormalization()
        self.relu_2_8 = layers.ReLU()

        self.cat_2 = layers.Concatenate()

        self.add = layers.Add()
        self.pool = layers.MaxPool2D(strides=2)

    def call(self, inputs, trainable=False):
        x_1 = self.conv_1_1(inputs, trainable=trainable)
        x_2 = self.conv_1_2(inputs, trainable=trainable)
        x_4 = self.conv_1_4(inputs, trainable=trainable)
        x_8 = self.conv_1_8(inputs, trainable=trainable)
        x_1 = self.bn_1_1(x_1, trainable=trainable)
        x_2 = self.bn_1_2(x_2, trainable=trainable)
        x_4 = self.bn_1_4(x_4, trainable=trainable)
        x_8 = self.bn_1_8(x_8, trainable=trainable)
        x_1 = self.relu_1_1(x_1, trainable=trainable)
        x_2 = self.relu_1_2(x_2, trainable=trainable)
        x_4 = self.relu_1_4(x_4, trainable=trainable)
        x_8 = self.relu_1_8(x_8, trainable=trainable)
        x = self.cat([x_1, x_2, x_4, x_8])

        x_1 = self.conv_2_1(x, trainable=trainable)
        x_2 = self.conv_2_2(x, trainable=trainable)
        x_4 = self.conv_2_4(x, trainable=trainable)
        x_8 = self.conv_2_8(x, trainable=trainable)
        x_1 = self.bn_2_1(x_1, trainable=trainable)
        x_2 = self.bn_2_2(x_2, trainable=trainable)
        x_4 = self.bn_2_4(x_4, trainable=trainable)
        x_8 = self.bn_2_8(x_8, trainable=trainable)
        x_1 = self.relu_2_1(x_1, trainable=trainable)
        x_2 = self.relu_2_2(x_2, trainable=trainable)
        x_4 = self.relu_2_4(x_4, trainable=trainable)
        x_8 = self.relu_2_8(x_8, trainable=trainable)
        x = self.cat([x_1, x_2, x_4, x_8])

        x = self.add(inputs, x)
        x = self.pool(x)

        return x

class ModResNet18Base(tf.keras.model):
    def __init__(self):
        super().__init__()
        self.block_1 = ModResNetBlock(32)
        self.block_2 = ModResNetBlock(32)
        self.block_3 = ModResNetBlock(64)
        self.block_4 = ModResNetBlock(64)
        self.block_5 = ModResNetBlock(128)
        self.block_6 = ModResNetBlock(128)
        self.block_7 = ModResNetBlock(256)
        self.block_8 = ModResNetBlock(256)
        self.block_9 = ModResNetBlock(512)

    def call(self, inputs, trainable=False):
        x = self.block1(inputs, trainable=trainable)
        x = self.block2(x, trainable=trainable)
        x = self.block3(x, trainable=trainable)
        x = self.block4(x, trainable=trainable)
        x = self.block5(x, trainable=trainable)
        x = self.block6(x, trainable=trainable)
        x = self.block7(x, trainable=trainable)
        x = self.block8(x, trainable=trainable)
        x = self.block9(x, trainable=trainable)
        return x

class ModResNet18Classify(ModResNet18Base):
    def __init__(self):
        super().__init__()
        self.dense_1 = layers.Dense(1000, activation='relu')
        self.dense_2 = layers.Dense(1000, activation='softmax')

    def call(self, inputs, trainable=False):
        x = super().call(inputs, trainable)
        x = self.dense_1(x, trainable=trainable)
        x = self.dense_2(x, trainable=trainable)
        return x

    
class ModResNet18FCN(ModResNet18Base):
    def __init__(self):
        super().__init__()
        pass

    def call(self, inputs, trainable=False):
        pass

    
if __name__ == '__main__':
    pass
