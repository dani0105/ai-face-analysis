from modules import light_cnn,celeb_gen
import numpy as np             

extractor_path = "lightcnn-face-recognition/model/lcnn_extract29v2_lr0.00100_loss2.689_valacc0.867_epoch0030.hdf5"
classifier_path = "lightcnn-face-recognition/model/lcnn_classify_lr0.00100_loss2.689_valacc0.867_epoch0030.hdf5"

datagen = celeb_gen.Datagen('lightcnn-face-recognition/data/output')
lcnn = light_cnn.LightCNN(classes=datagen.get_classes(),
                extractor_weights=extractor_path,
                classifier_weights=classifier_path)


gen = datagen.get_generator('test', batch_size=10)
x, y = next(gen)
y = np.argmax(y, axis=-1)

y_pred = lcnn.predict_class(x)

print('y')
print(y)

print('y_pred')
print(y_pred)