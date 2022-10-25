from tensorflow.keras.optimizers import SGD
from modules.celeb_gen import Datagen
from modules.light_cnn import LightCNN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# configuration
dataset = "lightcnn-face-recognition/data/output"
output = "lightcnn-face-recognition/model/lcnn_"
epochs = 2
steps_per_epoch = 2
out_period = 2

# dataset and model
datagen = Datagen(dataset)
lcnn = LightCNN(classes=datagen.get_classes())

train_gen = datagen.get_generator('train', batch_size=64)
valid_gen = datagen.get_generator('val', batch_size=64)

# training
lcnn.train(train_gen=train_gen, 
          valid_gen=valid_gen,
          optimizer=
            SGD(lr=0.001, momentum=0.9, decay=0.00004, nesterov=True),
          classifier_dropout=0.7, 
          steps_per_epoch=steps_per_epoch, 
          validation_steps=2,
          epochs=epochs, 
          out_prefix=output, 
          out_period=out_period)