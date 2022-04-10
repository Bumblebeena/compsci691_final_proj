import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class ResNetBlock(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=3, padding='same', trainable=False):
        super().__init__()
        self.conv_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, trainable=trainable)
        self.bn_1   = layers.BatchNormalization(trainable=trainable)
        self.relu_1 = layers.ReLU()
        self.conv_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, trainable=trainable)
        self.bn_2   = layers.BatchNormalization(trainable=trainable)
        self.add    = layers.Add()
        self.relu_2 = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.add(inputs, x)
        x = self.relu_2(x)
        return x

class ResnNetBlockDownSample(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=3, padding='same', trainable=False):
        super().__init__()
        self.conv_1_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, strides=(2,2), trainable=trainable)
        self.bn_1_1   = layers.BatchNormalization(trainable=trainable)
        self.relu_1_1 = layers.ReLU()
        self.conv_1_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, strides=(1,1), trainable=trainable)
        self.bn_1_2   = layers.BatchNormalization(trainable=trainable)
        self.conv_2_1 = layers.Conv2D(num_filters, kernel_size=1, padding='valid', strides=(2,2), trainable=trainable)
        self.bn_2_1   = layers.BatchNormalization(trainable=trainable)
        self.add      = layers.Add()
        self.relu_2_2 = layers.ReLU()
    
class ResNet18Base(tf.keras.Model):
    def __init__(self, trainable=False):
        super().__init__()
        self.conv_a  = layers.Conv3d(kernels = 64, kernel_size=7, trainable=trainable)
        self.bn_a    = layers.BatchNormalization()
        self.relu_a  = layers.ReLU()
        self.pool_a  = layers.MaxPool2D(strides=2)
        self.block_1 = ResNetBlock(64, trainable=trainable)
        self.block_2 = ResNetBlock(64, trainable=trainable)
        self.block_3 = ResNetBlockDownSample(128, trainable=trainable)
        self.block_4 = ResNetBlock(128, trainable=trainable)
        self.block_5 = ResNetBlock(128, trainable=trainable)
        self.block_6 = ResNetBlock(128, trainable=trainable)
        self.block_7 = ResNetBlock(256, trainable=trainable)
        self.block_8 = ResNetBlock(256, trainable=trainable)
        self.block_9 = ResNetBlock(512, trainable=trainable)
        
    def call(self, inputs, training=False):
        x = self.conv_a(inputs)
        x = self.bn_a(x)
        x = self.pool_a(x)
        x = self.block_1(x, training)
        x = self.block_2(x, training)
        x = self.block_3(x, training)
        x = self.block_4(x, training)
        x = self.block_5(x, training)
        x = self.block_6(x, training)
        x = self.block_7(x, training)
        x = self.block_8(x, training)
        x = self.block_9(x, training)
        return x

class ResNet18Classify(ResNet18Base):
    def __init__(self, trainable=False):
        super().__init__(trainable=trainable)
        self.flat_1 = layers.Flatten()
        self.dense_1 = layers.Dense(512, activation='relu')
        self.drop_3 = layers.Dropout(0.2)
        self.dense_2 = layers.Dense(10, activation='softmax')
        

    def call(self, inputs, training=False):
        x = super().call(inputs, training)
        x = self.flat_1(x)
        x = self.dense_1(x)
        if (training):
            x = self.drop_3(x)
        x = self.dense_2(x)
        return x

class ResNet18FCN(ResNet18Base):
    def __init__(self):
        super().__init__()
        pass

    def call(self, inputs, training=False):
        pass

if __name__ == '__main__':
    net = ResNet18Classify()
