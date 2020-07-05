import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report

BATCH_SIZE = 20
NUM_CLASSES = 11


def csv_feature_extraction(path, batch_size):
    file = open(path, "r")
    while True:
        data = []
        while len(data) < batch_size:
            row = file.readline()
            if row == '':
                file.seek(0)
                row = file.readline()
            row = row.strip().split(',')
            features = np.array(row[2:], dtype='float')
            data.append(features)
        yield np.array(data)


print('[INFO] Extracting the information')
fileNameList = [element.split(',')[0] for element in open('test_feature.csv')]
labelList = [int(element.split(',')[1]) for element in open('test_feature.csv')]
print(labelList[:20])
totalTest = len(labelList)
print('[INFO] Loading model')
model = keras.models.load_model('CNN_Model.h5')
testGen = csv_feature_extraction('test_feature.csv', BATCH_SIZE)
print('[INFO] Predicting model')
predict = model.predict(x=testGen, steps=(totalTest // BATCH_SIZE), verbose=1)
predicted_class_indices = np.argmax(predict, axis=1)
print(classification_report(labelList, predicted_class_indices))
print('[INFO] Writing file')
csv = open('predict.csv', "w")
csv.write('{},{}\n'.format('filename','predict category'))
for i in range (totalTest):
    fileName = fileNameList[i]
    predictCategory = predicted_class_indices[i]
    csv.write('{},{}\n'.format(fileName,predictCategory))
csv.close()
