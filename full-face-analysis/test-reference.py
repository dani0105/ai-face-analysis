import numpy as np
from keras.preprocessing.image import image as keras_image
from keras.preprocessing.image import ImageDataGenerator
from modules import light_cnn,celeb_gen


process = ImageDataGenerator()

img = keras_image.load_img("test.jpg",
        color_mode='grayscale',
        target_size=(144,144),
        interpolation='nearest')
x = keras_image.img_to_array(img, data_format='channels_last')

#params = process.get_random_transform(x.shape)
#x = process.apply_transform(x, params)
#x = process.standardize(x)
batch_x = np.zeros((1,) + (144,144,1), dtype='float32')
batch_x[0] = x

base_offset = 144 - 128
h_begin = int(base_offset*np.random.random())
h_end = h_begin + 128
w_begin = int(base_offset*np.random.random())
w_end = w_begin + 128
batch_x = batch_x[:, h_begin:h_end, w_begin:w_end, :]
batch_x /= 255.


extractor_path = "lightcnn-face-recognition/model/lcnn_extract29v2_lr0.00100_loss2.689_valacc0.867_epoch0030.hdf5"
classifier_path = "lightcnn-face-recognition/model/lcnn_classify_lr0.00100_loss2.689_valacc0.867_epoch0030.hdf5"

datagen = celeb_gen.Datagen('lightcnn-face-recognition/data/output')
lcnn = light_cnn.LightCNN(classes=datagen.get_classes(),
                extractor_weights=extractor_path,
                classifier_weights=classifier_path)
                
y_pred = lcnn.predict_class(batch_x)

print(y_pred)