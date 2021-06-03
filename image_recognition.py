#Image recognition using tensorflow

import numpy as np
import pandas as pd
import os
import cv2
from random2 import sample

import matplotlib.pyplot as plt
import seaborn as sns
from scikitplot.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from tensorflow.compat.v1.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model

data = pd.read_csv("full_df.csv")
data.head()


#extracting data of cataract symptopms and normal only from the metadata
def prepare(data):
    leftdf = data[(data['Left-Diagnostic Keywords'].str.contains('cataract')) & (data['C'] == 1 )][['C','Left-Fundus']] #cataract left eye
    leftdf = leftdf.append(data[(data['Left-Diagnostic Keywords'].str.contains('normal fundus')) & (data['C'] == 0 )][['C','Left-Fundus']]).rename(columns={"Left-Fundus": "filename"}) #normal left eye
    rightdf = data[(data['Right-Diagnostic Keywords'].str.contains('cataract')) & (data['C'] == 1 )][['C','Right-Fundus']] #cataract right eye
    rightdf = rightdf.append(data[(data['Right-Diagnostic Keywords'].str.contains('normal fundus')) & (data['C'] == 0 )])[['C','Right-Fundus']].rename(columns={"Right-Fundus": "filename"}) #normal right eye
    prepareddf = leftdf.append(rightdf) 
    return prepareddf
prepareddf = prepare(data)
len(prepareddf)


sns.set(font_scale=1.4)
prepareddf['C'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0, color = ['r', 'g'])
plt.xticks([0, 1], ['Cataract', 'Normal'])
plt.ylabel("Count of People", labelpad=14)
plt.title("Count of People With Cataract and Normal", y=1.02);


#to reduce the computational time, just take the C and filename columns
target = prepareddf['C'].to_list()
filenames = prepareddf['filename'].to_list()


#import and prepare the image data
folder = "preprocessed_images"
height = 256
width = 256

def preimage(filenames, target):
    imagedata = []
    for idx, imagename in enumerate(filenames):

        image = cv2.imread(os.path.join(folder,imagename))
        try:
            image = cv2.resize(image, (width, height))
            imagedata.append(image)
        except:
            del target[idx] #delete the row/index of the inexcutable image file
    imagedata = np.array(imagedata)
    return imagedata, target

imagedata, target = preimage(filenames, target)

print(imagedata.shape)


#display the data that we are dealing with
f, ax = plt.subplots(3, 3, figsize=(10,10))

target_9 = target[:9]
imagedata_9 = imagedata[:9]

for idx, image in enumerate (imagedata_9):
    ax[idx//3, idx%3].imshow(image)
    ax[idx//3, idx%3].axis('off')
    if target_9[idx] == 0:
        ax[idx//3, idx%3].set_title('Normal')
    else:
        ax[idx//3, idx%3].set_title('Cataract')
plt.show()


#create balanced data for the x_data (independent variables) and y_data(dependent variables)
cat_count = target.count(1) #store the number of cataract cases

def balancingxy(target, imagedata):
    #generating index of the normal cases
    normal_idx = []

    for idx, diagnosis in enumerate (target):
        if diagnosis == 0:
            normal_idx.append(idx)

    normal_idx_rd = sample(normal_idx, cat_count)

    #generating index of the cataract cases
    cat_idx = []

    for idx, diagnosis in enumerate (target):
        if diagnosis == 1:
            cat_idx.append(idx)
            
    x_data = []
    y_data = []

    idx_data = normal_idx_rd + cat_idx
    for idx in idx_data:
        x_data.append(imagedata[idx])
        y_data.append(target[idx])
        
    return x_data, y_data

x_data, y_data = balancingxy(target, imagedata)


x_data = np.array(x_data)
y_data = np.array(y_data)
y_data = np.expand_dims(y_data, axis = -1) #change the dimension by inserting an axis into the last position
y_data = tf.keras.utils.to_categorical(y_data) #transofrming the target variable itu dummy variable

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state = 123)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#transfer learning using ResNet50
resnet50 = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(256, 256, 3)))

for layer in resnet50.layers:
    layer.trainable = False #freeze the layer

    
#building the model
model = Sequential()
model.add(resnet50)
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(2,activation = "sigmoid"))

model.summary()


#set up earlystopping and reduce learning rate when metric stops improving
callback = tf.keras.callbacks.EarlyStopping(patience=20,                                                verbose=1, 
                                               restore_best_weights=True)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, verbose=1)

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy', 'Recall', 'Precision'])

hist = model.fit(x_train, y_train, 
                    validation_data = (x_test, y_test), 
                    epochs = 10,
                    batch_size = 64,
                    callbacks=[callback, reducelr])


#plot model accuracy and loss
sns.set()
fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(hist.epoch, hist.history['accuracy'], label = 'training')
sns.lineplot(hist.epoch, hist.history['val_accuracy'], label = 'validation')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(hist.epoch, hist.history['loss'], label = 'training')
sns.lineplot(hist.epoch, hist.history['val_loss'], label = 'validation')
plt.title('Loss')
plt.tight_layout()

plt.show()


#evaluate model using test data
model.evaluate(x_test, y_test)

prediction = model.predict_classes(x_test)
y_observed = np.argmax(y_test, axis=1) #retrieve back the target variable as one dim array

plot_confusion_matrix(y_observed, prediction, figsize=(14,14))
plt.show()

