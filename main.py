import ResNet18 as rs
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

data, info = tfds.load('cifar10', split=['train', 'test'], with_info=True)
train = data[0]
test = data[1]
train_batch = train.batch(100)

net = rs.ResNet18Classify(True)
net.compile(optimizer="Adam", loss="mse")


net.fit(train_batch)
