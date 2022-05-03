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
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from PARCnet import PARCnetClassify
from Utils import normalize_and_resize_img


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    num_classes = 1000
    batch_size = 128
    num_feature_maps = 512
    trainable = True
    model = PARCnetClassify(num_classes, batch_size, num_feature_maps, trainable)

    [ds_train, ds_test, ds_val], ds_info = tfds.load('imagenet2012_subset/10pct', split=['train[:10%]', 'validation[:5%]', 'validation[5%:10%]'], shuffle_files=False, as_supervised=True, with_info=True)

    num_train = ds_info.splits['train[:5%]'].num_examples
    num_val = ds_info.splits['validation[:2%]'].num_examples
    num_test = ds_info.splits['train[5:7%]'].num_examples

    ds_train = ds_train.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(num_train)
    ds_train = ds_train.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # one = ds_train.batch(1).take(1)
    # for image, label in one:
    #     print(image[0].shape)
    #     out = model.predict(image[0])
    #     print(out.shape)
    #     print(out.argmax(axis=-1))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    checkpoint_path = "./training{}".format(timestamp)
    checkpoint_path += "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        save_freq='epoch')

    csv_log_path = 'train_log_{}.csv'.format(timestamp)
    csv_logger = CSVLogger(csv_log_path, append=True, separator=',')
    time_callback = TimeHistory()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy',tf.keras.metrics.OneHotMeanIoU(num_classes=11)])

    model.save_weights(checkpoint_path.format(epoch=0))    

    # TODO: make sure this is how you access images and labels
    model.fit(x=ds_train['images'],
              y=ds_train['labels'],
              batch_size=batch_size,
              epochs=2,
              verbose=1,
              callbacks = [cp_callback,csv_logger,time_callback],
              validation_data=(ds_val['images'],ds_val['labels'],
              shuffle=True
              validation_steps=batch_size
              )

    with open(csv_log_path, 'a') as csv_out:
        pass
              
