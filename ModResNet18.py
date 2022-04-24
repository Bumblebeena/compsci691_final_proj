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

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt



class ModResNetBlock(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=3, padding='same', trainable=False, cat=True):
        super().__init__()
        num_filters /= 4
        self.conv_1_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(1,1), trainable=trainable)
        self.relu_1_1 = layers.ReLU()
        self.bn_1_1 = layers.BatchNormalization()
        self.drop_1_1 = layers.Dropout(0.05)

        self.conv_1_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(2,2), trainable=trainable)
        self.relu_1_2 = layers.ReLU()
        self.bn_1_2 = layers.BatchNormalization()
        self.drop_1_2 = layers.Dropout(0.05)

        self.conv_1_4 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(4,4), trainable=trainable)
        self.relu_1_4 = layers.ReLU()
        self.bn_1_4 = layers.BatchNormalization()
        self.drop_1_4 = layers.Dropout(0.05)

        self.conv_1_8 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(8,8), trainable=trainable)
        self.relu_1_8 = layers.ReLU()
        self.bn_1_8 = layers.BatchNormalization()
        self.drop_1_8 = layers.Dropout(0.05)
        self.cat_1 = layers.Concatenate()
        
        self.conv_2_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(1,1), trainable=trainable)
        self.relu_2_1 = layers.ReLU()
        self.bn_2_1 = layers.BatchNormalization()
        self.drop_2_1 = layers.Dropout(0.05)
        
        self.conv_2_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(2,2), trainable=trainable)
        self.relu_2_2 = layers.ReLU()
        self.bn_2_2 = layers.BatchNormalization()
        self.drop_2_2 = layers.Dropout(0.05)

        self.conv_2_4 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(4,4), trainable=trainable)
        self.relu_2_4 = layers.ReLU()
        self.bn_2_4 = layers.BatchNormalization()
        self.drop_2_4 = layers.Dropout(0.05)

        self.conv_2_8 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(8,8), trainable=trainable)
        self.relu_2_8 = layers.ReLU()
        self.bn_2_8 = layers.BatchNormalization()
        self.drop_2_8 = layers.Dropout(0.05)
        
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
    def __init__(self, num_filters, kernel_size=3, padding='same', trainable=False):
        super().__init__()
        num_filters /= 4
        self.conv_1_1 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(2,2), trainable=trainable)
        self.relu_1_1 = layers.ReLU()
        self.bn_1_1 = layers.BatchNormalization()
        self.drop_1_1 = layers.Dropout(0.05)

        self.conv_1_2 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(4,4), trainable=trainable)
        self.relu_1_2 = layers.ReLU()
        self.bn_1_2 = layers.BatchNormalization()
        self.drop_1_2 = layers.Dropout(0.05)

        self.conv_1_4 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(8,8), trainable=trainable)
        self.relu_1_4 = layers.ReLU()
        self.bn_1_4 = layers.BatchNormalization()
        self.drop_1_4 = layers.Dropout(0.05)

        self.conv_1_8 = layers.Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=(16,16), trainable=trainable)
        self.relu_1_8 = layers.ReLU()
        self.bn_1_8 = layers.BatchNormalization()
        self.drop_1_8 = layers.Dropout(0.05)
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



class ModResNet18Base(tf.keras.Model):
    def __init__(self, trainable=False):
        super().__init__()
        # self.conv_spread = ConvSpread(32, trainable=trainable)
        # self.block_1 = ModResNetBlock(32, trainable=trainable, cat=True)
        # self.block_2 = ModResNetBlock(64, trainable=trainable, cat=False)
        # self.block_3 = ModResNetBlock(64, trainable=trainable, cat=True)
        # self.block_4 = ModResNetBlock(128, trainable=trainable, cat=False)
        # self.block_5 = ModResNetBlock(128, trainable=trainable, cat=True)
        # self.block_6 = ModResNetBlock(256, trainable=trainable, cat=False)
        # self.block_7 = ModResNetBlock(256, trainable=trainable, cat=True)
        # self.block_8 = ModResNetBlock(512, trainable=trainable, cat=False)
        # self.block_9 = ModResNetBlock(512, trainable=trainable, cat=False)

        self.block_1 = ModResNetBlock(64, trainable=trainable, cat=True)
        self.block_2 = ModResNetBlock(128, trainable=trainable, cat=True)
        self.block_3 = ModResNetBlock(256, trainable=trainable, cat=True)
        self.block_4 = ModResNetBlock(512, trainable=trainable, cat=True)
        self.block_5 = ConvSpread(1024, trainable=trainable)


        self.pool_1 = layers.MaxPool2D(strides=2)
        self.pool_2 = layers.MaxPool2D(strides=2)
        self.pool_3 = layers.MaxPool2D(strides=2)
        self.pool_4 = layers.MaxPool2D(strides=2)
        self.pool_5 = layers.MaxPool2D(strides=2)


    def call(self, inputs, trainable=False):
        # x = self.conv_spread(inputs)
        # x = self.pool_1(x)
        # x = self.block_1(x)
        # x = self.block_2(x)
        # x = self.block_3(x)
        # x = self.pool_2(x)
        # x = self.block_4(x)
        # x = self.block_5(x)
        # x = self.pool_3(x)
        # x = self.block_6(x)
        # x = self.pool_4(x)
        # x = self.block_7(x)
        # x = self.pool_5(x)
        # x = self.block_8(x)

        x = self.block_1(inputs)
        x = self.pool_1(x)
        x = self.block_2(x)
        x = self.pool_2(x)
        x = self.block_3(x)
        x = self.pool_3(x)
        x = self.block_4(x)
        x = self.pool_4(x)
        x = self.block_5(x)

        return x

class ModResNet18Classify(ModResNet18Base):
    def __init__(self, batch_size=256, trainable=False):
        super().__init__(trainable=trainable)
        self.pool_g  = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(1000, input_shape=(batch_size,1024), activation='softmax')


    def call(self, inputs, trainable=False):
        x = super().call(inputs)
        x = self.pool_g(x)
        x = self.dense_1(x)
        return x

    
class ModResNet18FCN(ModResNet18Base):
    def __init__(self):
        super().__init__()
        pass

    def call(self, inputs, trainable=False):
        pass

def normalize_and_resize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return tf.image.resize(image, [224,224]), label


if __name__ == '__main__':
    batch_size = 10
    model = ModResNet18Classify(batch_size, trainable=True)
    model.build((batch_size,224,224,3))
    model.summary()
    
    [ds_train, ds_test, ds_val], ds_info = tfds.load('imagenette', split=['train[:5%]', 'train[5%:7%]', 'validation[:5%]'], shuffle_files=False, as_supervised=True, with_info=True)


    num_train = ds_info.splits['train[:5%]'].num_examples
    ds_train = ds_train.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(num_train)
    ds_train = ds_train.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    num_valid = ds_info.splits['validation[:5%]'].num_examples
    ds_val = ds_val.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    num_test = ds_info.splits['train[5:7%]'].num_examples
    ds_test = ds_test.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            batch_loss = loss_fn(y, logits)

        gradients = tape.gradient(batch_loss, model.trainable_weights)

        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        correct = tf.math.count_nonzero(tf.equal(tf.math.argmax(logits, 1), tf.math.argmax(y,1)))

        return batch_loss, correct

    @tf.function
    def test_step(x, y):
        test_logits = model(x, training=False)
        test_loss = loss_fn(y, test_logits)
        test_correct = tf.math.count_nonzero(tf.equal(tf.math.argmax(test_logits, 1), tf.math.argmax(y, 1)))
        return test_loss, test_correct
    
    num_epochs = 10

    num_batches = int(num_train / batch_size)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        for x_i, y_i in ds_train:
            batch_loss, batch_correct = train_step(x_i, y_i)
            epoch_correct += batch_correct.numpy()
            epoch_loss += batch_loss

        train_loss.append(epoch_loss/num_batches)
        train_acc.append(epoch_correct/num_train)

        val_loss_sum = 0
        val_correct = 0
        for x_valid, y_valid in ds_val:
            batch_val_loss, batch_val_correct = test_step(x_valid, y_valid)
            val_loss_sum += batch_val_loss
            val_correct += batch_val_correct.numpy()
            
        val_loss.append(val_loss_sum/num_batches)
        val_acc.append(val_correct/num_valid)
        print("Completed epoch {} of {}".format(epoch+1, num_epochs))


    # # Run our test set through the trained model
    # x_test, y_test = 
    # test_acc = np.ones(num_epochs)
    # _, test_correct = test_step(x_test, y_test)
    # test_acc *= (test_correct/num_test)


    # Generate plots according to project requirements and save them to the
    # output folder, which is guaranteed at this point to exist
    epochs = np.arange(num_epochs)

    plt.plot(epochs, np.array(train_acc)*100, label="train")
    plt.plot(epochs, np.array(val_acc)*100, label="valid")
    #plt.plot(epochs, test_acc*100, label="test")
    plt.title("PARCnet Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy's")
    plt.legend()
    plt.grid()
    plt.savefig("output/accuracy.png")
    plt.close()

    plt.plot(epochs, np.array(train_loss), label="train")
    plt.plot(epochs, np.array(val_loss), label="valid")
    plt.title("PARCnet Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss's")
    plt.legend()
    plt.grid()
    plt.savefig("output/loss.png")
    plt.close()
