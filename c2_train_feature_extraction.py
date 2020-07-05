from typing import BinaryIO
import os
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
from imutils import paths
import random
from sklearn.preprocessing import LabelEncoder
import pickle

BATCH_SIZE = 32
list_train = []
csv = open('train_feature.csv', "w")
imagePaths = list(paths.list_images('train_2'))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
print(labels)
le = LabelEncoder()
le.fit(labels)
totalBatches = int(np.ceil(len(imagePaths) / BATCH_SIZE))
model = ResNet50(weights='imagenet', include_top=False)
for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
    print('[INFO] Processing batch {} / {}'.format(b+1, totalBatches))
    batchPaths = imagePaths[i:i+BATCH_SIZE]
    batchLabels = le.transform(labels[i:i+BATCH_SIZE])
    batchImages = []
    for imagePath in batchPaths:
        img = image.load_img(imagePath, target_size=(224, 224))
        imgData = image.img_to_array(img)
        imgData = np.expand_dims(imgData, axis=0)
        imgData = preprocess_input(imgData)
        batchImages.append(imgData)
    batchImages = np.vstack(batchImages)
    resNet50Features = model.predict(batchImages, batch_size=BATCH_SIZE)
    resNet50Features = resNet50Features.reshape((resNet50Features.shape[0], 7 * 7 * 2048))
    for (label, feature) in zip(batchLabels, resNet50Features):
        featureJoin = ','.join([str(v) for v in feature])
        csv.write("{},{}\n".format(label, featureJoin))
csv.close()
f: BinaryIO = open('label_encoder', "wb")
f.write(pickle.dumps(le))
f.close()


