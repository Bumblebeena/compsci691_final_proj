import tensorflow as tf
from tensorflow.keras.callbacks import Callback



#####################################################################
# from https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
#####################################################################

def normalize_and_resize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return tf.image.resize(image, [224,224]), label

def normalize(image, label):
    return tf.cast(image, tf.float32) / 255., label

def one_hot(x, num_classes=11):
    return tf.one_hot(x, num_classes)
