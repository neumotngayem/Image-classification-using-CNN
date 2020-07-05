from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

BATCH_SIZE = 32
NUM_CLASSES = 11


def csv_feature_extraction(path, batch_size, num_classes):
    file = open(path, "r")
    while True:
        data = []
        labels = []
        while len(data) < batch_size:
            row = file.readline()
            if row == '':
                file.seek(0)
                row = file.readline()
            row = row.strip().split(',')
            label = row[0]
            category = to_categorical(label, num_classes=num_classes)
            features = np.array(row[1:], dtype='float')
            labels.append(category)
            data.append(features)
        yield (np.array(data), np.array(labels))


totalTrain = 16992
print('[INFO] Assign generator function')
trainGen = csv_feature_extraction('train_feature.csv', BATCH_SIZE, NUM_CLASSES)
print('[INFO] Create CNN model architecture')
model = Sequential()
model.add(Dense(1500, input_shape=(7 * 7 * 2048,), activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
print('[INFO] Compile ANN model architecture')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('[INFO] Fit ANN model architecture')
model.fit(x=trainGen, steps_per_epoch=totalTrain // BATCH_SIZE, epochs=5)
print('[INFO] Save ANN model architecture')
model.save("CNN_Model.h5")
