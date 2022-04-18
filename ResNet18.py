import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.backend import constant as k
import numpy as np
import cv2

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
        self.conv_a  = layers.Conv2D(filters=64, kernel_size=7, trainable=trainable)
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

class ResNet18Classify(ResNet18Base):
    def __init__(self, batch_size=256, trainable=False):
        super().__init__(trainable=trainable)
        self.pool_1 = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(1000, input_shape=(batch_size,512), activation='softmax')
        

    def call(self, inputs, training=False):
        x = super().call(inputs)
        x = self.pool_1(x)
        x = self.dense_1(x)
        return x

class ResNet18FCN(ResNet18Base):
    def __init__(self):
        super().__init__()
        pass

    def call(self, inputs, training=False):
        pass

def normalize_and_resize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return tf.image.resize(image, [224,224]), label

if __name__ == '__main__':
    batch_size = 10
    model = ResNet18Classify(batch_size, trainable=True)
    model.build((batch_size,224,224,3))
    model.summary()

     
    # img = cv2.imread("/Users/djohnston/umbc/computer_vision/final_proj/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
    # img_resize = cv2.resize(img, (224,224),)
    # print(img_resize.shape())
    # model.predict(k((img_resize)))
    
    [ds_train, ds_test, ds_val], ds_info = tfds.load('imagenette', split=['train[:75%]', 'train[75%:]', 'validation'], shuffle_files=False, as_supervised=True, with_info=True)
    #print(ds_train)
    ds_train = ds_train.map(normalize_and_resize_img, num_parallel_calls=tf.data.AUTOTUNE)

    
    # print(dir(one))
    # for image, label in one:
    #     print(image.shape)
    #     print(np.max(model.predict(image)))

    num_train = ds_info.splits['train'].num_examples
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(num_train)
    ds_train = ds_train.batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=1000)))
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    for a, b in ds_train.take(1):
        print(b)
    #     model.predict(cv2.resize(ex))

    # ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = ds_test.batch(256)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    
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
    
    num_epochs = 1

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
            epoch_loss = batch_loss

        train_loss.append(epoch_loss)
        train_acc.append(epoch_correct/num_train)

        epoch_val_loss, epoch_val_correct = test_step(x_valid, y_valid)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_correct/num_valid)
        print("Completed epoch {} of {}".format(epoch+1, num_epochs))


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
