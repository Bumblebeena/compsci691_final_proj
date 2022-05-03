import os
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin')
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/lib64')
os.add_dll_directory(r'C:/Program Files/NVIDIA/CUDNN/v8.3/bin')
os.add_dll_directory(r'C:/Program Files/zlib/dll_x64')

from PARCnet import PARCnetSeg
from ResNet18 import ResNet18Seg
import VOC2012 as vc
import numpy as np
import cv2



if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    batch_size = 1

    # model = ModResNet18FcnFullRes(trainable=True)
    # model.build((batch_size,224,224,3))
    # model.summary()
    voc_base_dir = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    voc = vc.VOC2012(voc_base_dir, resize_method='pad', checkpaths=True)

    voc.load_test_data('./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/voc2012_224_test')

    voc.test_images = np.expand_dims(np.array(voc.test_images), 1)
    voc.test_labels = np.expand_dims(np.array(voc.test_labels), 1)
    a = voc.test_images[10]
    b = voc.test_labels[10]
    print(a.shape)

    for i in range(20,100):
        #model = PARCnetEncoderDecoder
        model = ResNet18Seg(11)
        model.built = True
        model.load_weights('training_ResNet18_2022-05-01_15-44/cp-00{}.ckpt.index'.format(i))

        out = model.predict(a)
        #print(out.shape)
        out_img = out.argmax(axis=-1)
        print(out_img)

        # cv2.imwrite('test1.png', a[0,:,:])
        # cv2.imwrite('test2.png', voc.gray_to_rgb(b[0,:,:]))
        cv2.imwrite('testimg_resnet/test{}.png'.format(i), voc.gray_to_rgb(out_img[0,:,:]))
    
