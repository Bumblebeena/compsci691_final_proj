import os
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin')
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/lib64')
os.add_dll_directory(r'C:/Program Files/NVIDIA/CUDNN/v8.3/bin')
os.add_dll_directory(r'C:/Program Files/zlib/dll_x64')

import csv
import time
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, callbacks
from tensorflow.keras.backend import constant as k
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
import cv2

import ctypes

import VOC2012 as vc

   



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
        x = self.add([inputs, x])
        x = self.relu_2(x)
        return x

class ResNetBlockDownSample(tf.keras.Model):
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

    def call(self, inputs, training=False):
        x1 = self.conv_1_1(inputs)
        x1 = self.bn_1_1(x1)
        x1 = self.relu_1_1(x1)
        x1 = self.conv_1_2(x1)
        x1 = self.bn_1_2(x1)

        x2 = self.conv_2_1(inputs)
        x2 = self.bn_2_1(x2)
        x  = self.add([x1, x2])
        x  = self.relu_2_2(x)
        return x
        
    
class ResNet18Base(tf.keras.Model):
    def __init__(self, trainable=False):
        super().__init__()
        self.conv_a  = layers.Conv2D(filters=64, kernel_size=7, padding='same', trainable=trainable)
        self.bn_a    = layers.BatchNormalization()
        self.relu_a  = layers.ReLU()
        self.pool_a  = layers.MaxPool2D(strides=2)
        self.block_1 = ResNetBlock(64, trainable=trainable)
        self.block_2 = ResNetBlock(64, trainable=trainable)
        self.block_3 = ResNetBlockDownSample(128, trainable=trainable)
        self.block_4 = ResNetBlock(128, trainable=trainable)
        self.block_5 = ResNetBlockDownSample(256, trainable=trainable)
        self.block_6 = ResNetBlock(256, trainable=trainable)
        self.block_7 = ResNetBlockDownSample(512, trainable=trainable)
        self.block_8 = ResNetBlock(512, trainable=trainable)
        
    def call(self, inputs, training=False):
        x = self.conv_a(inputs)
        x = self.bn_a(x)
        x = self.pool_a(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)

        return x

class ResNet18Seg(ResNet18Base):
    def __init__(self, classes, trainable=False):
        super().__init__(trainable)
        kernel_size = 3
        padding = 'same'

        self.conv_1 = layers.Conv2D(256, kernel_size, padding=padding, activation='relu', trainable=trainable)
        self.conv_2 = layers.Conv2D(128, kernel_size, padding=padding, activation='relu', trainable=trainable)
        self.conv_3 = layers.Conv2D(64, kernel_size, padding=padding, activation='relu', trainable=trainable)
        self.conv_4 = layers.Conv2D(64, kernel_size, padding=padding, activation='relu', trainable=trainable)

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
        x = self.conv_a(inputs)
        a = self.bn_a(x)
        x = self.pool_a(a)
        x = self.block_1(x)
        b = self.block_2(x)
        x = self.block_3(b)
        c = self.block_4(x)
        x = self.block_5(c)
        d = self.block_6(x)
        x = self.block_7(d)
        x = self.block_8(x)

        x = self.up_1(x)
        x = self.cat_1([x,d])
        x = self.conv_1(x)
        x = self.up_2(x)
        x = self.cat_2([x,c])
        x = self.conv_2(x)
        x = self.up_3(x)
        x = self.cat_3([x,b])
        x = self.conv_3(x)
        x = self.up_4(x)
        x = self.cat_4([x,a])
        x = self.conv_4(x)

        x = self.squeeze(x)
        x = self.softmax(x)
        
        return x

class ResNet18Classify(ResNet18Base):
    def __init__(self, classes=11, batch_size=256, trainable=False):
        super().__init__(trainable=trainable)
        self.pool_1 = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(classes, input_shape=(batch_size,512), activation='softmax')
        

    def call(self, inputs, training=False):
        x = super().call(inputs)
        x = self.pool_1(x)
        x = self.dense_1(x)
        return x



class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def one_hot(x, num_classes=11):
    return tf.one_hot(x, num_classes)

def normalize_and_resize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return tf.image.resize(image, [224,224]), label

if __name__ == '__main__':
    batch_size = 10
    # model = ResNet18Classify(batch_size, trainable=True)
    model = ResNet18Seg(11, trainable=True)
    # model.build((batch_size,224,224,3))
    # model.summary()

     
    # img = cv2.imread("/Users/djohnston/umbc/computer_vision/final_proj/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
    # img_resize = cv2.resize(img, (224,224),)
    # print(img_resize.shape())
    # model.predict(k((img_resize)))
    
    # [ds_train, ds_test, ds_val], ds_info = tfds.load('imagenette', split=['train[:5%]', 'train[5%:7%]', 'validation[:5%]'], shuffle_files=False, as_supervised=True, with_info=True)
    #print(ds_train)
    # ds_train = ds_train.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)

    
    # print(dir(one))
    # for image, label in one:
    #     print(image.shape)
    #     print(np.max(model.predict(image)))

    # num_train = ds_info.splits['train'].num_examples
    # ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(num_train)
    # ds_train = ds_train.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    # ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # for a, b in ds_train.take(1):
    #     print(b)
    # #     model.predict(cv2.resize(ex))

    # ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = ds_test.batch(256)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    voc_base_dir = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    voc = vc.VOC2012(voc_base_dir, resize_method='resize', checkpaths=True)

    voc.load_train_data('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/voc2012_224_train')
    voc.load_test_data('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/voc2012_224_test')
    voc.load_val_data('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/voc2012_224_val')

    voc.convert_to_numpy()

    num_train = voc.train_images.shape[0]
    print('Num training images: {}'.format(num_train))
    num_test = voc.test_images.shape[0]
    print('Num test images: {}'.format(num_test))
    num_val = voc.val_images.shape[0]
    print('Num validation images: {}'.format(num_val))

    a, b = voc.get_batch_train(1)

    print(a.shape)
    print(a)
    out = model.predict(a)
    print(out.shape)
    print(out.argmax(axis=-1))


    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    checkpoint_path = "./training_ResNet18_{}".format(timestamp)
    checkpoint_path += "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     save_freq='epoch'
                                                     )

    csv_logger = CSVLogger('train_log_ResNet18_{}.csv'.format(timestamp), append=True, separator=',')

    time_callback = TimeHistory()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy', tf.keras.metrics.OneHotMeanIoU(num_classes=11)])

    model.save_weights(checkpoint_path.format(epoch=0))    

    model.fit(x=voc.train_images[:600],
              y=one_hot(voc.train_labels[:600]),
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              callbacks = [cp_callback,csv_logger,time_callback],
              shuffle=True,
              #validation_split=0.1,
              #validation_data=(voc.val_images[:50],one_hot(voc.val_labels[:50])),
              validation_steps=batch_size
              )


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
    
    # num_epochs = 1

    # num_batches = int(num_train / batch_size)

    # train_loss = []
    # val_loss = []
    # train_acc = []
    # val_acc = []
    # for epoch in range(num_epochs):
    #     epoch_loss = 0
    #     epoch_correct = 0
    #     for x_i, y_i in ds_train:
    #         batch_loss, batch_correct = train_step(x_i, y_i)
    #         epoch_correct += batch_correct.numpy()
    #         epoch_loss = batch_loss

    #     train_loss.append(epoch_loss)
    #     train_acc.append(epoch_correct/num_train)

    #     # epoch_val_loss, epoch_val_correct = test_step(x_valid, y_valid)
    #     # val_loss.append(epoch_val_loss)
    #     # val_acc.append(epoch_val_correct/num_valid)
    #     print("Completed epoch {} of {}".format(epoch+1, num_epochs))


    # # Run our test set through the trained model
    # x_test, y_test = 
    # test_acc = np.ones(num_epochs)
    # _, test_correct = test_step(x_test, y_test)
    # test_acc *= (test_correct/num_test)


    # # Generate plots according to project requirements and save them to the
    # # output folder, which is guaranteed at this point to exist
    # epochs = np.arange(num_epochs)

    # plt.plot(epochs, np.array(train_acc)*100, label="train")
    # plt.plot(epochs, np.array(val_acc)*100, label="valid")
    # plt.plot(epochs, test_acc*100, label="test")
    # plt.title("ResNet-4 Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy's")
    # plt.legend()
    # plt.grid()
    # plt.savefig("output/accuracy.png")
    # plt.close()

    # plt.plot(epochs, np.array(train_loss), label="train")
    # plt.plot(epochs, np.array(val_loss), label="valid")
    # plt.title("ResNet-4 Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss's")
    # plt.legend()
    # plt.grid()
    # plt.savefig("output/loss.png")
    # plt.close()
