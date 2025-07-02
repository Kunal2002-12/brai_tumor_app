import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.utils import to_categorical

input_size=64
image_dict = "F:/Data sc/deep learning projects/brain/datasets"

yes_tumor_images = os.listdir(os.path.join(image_dict, 'yes/'))
no_tumor_images = os.listdir(os.path.join(image_dict, 'no/'))
dataset=[]
label=[]

#print(no_tumor_images)

for image_name in no_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image_path = os.path.join(image_dict, 'no/', image_name)
        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image_path = os.path.join(image_dict, 'yes/', image_name)
        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(1)

#print(len(dataset))
#print(len(label))

dataset=np.array(dataset)
label=np.array(label)
x_train, x_test, y_train, y_test=train_test_split(dataset,label, test_size=0.2, random_state=0)

#Reshape=(n, image_width, image_height, n_channel)
# print("x_train:", x_train.shape)
# print("y_train:", y_train.shape)
# print("x_test:", x_test.shape)
# print("y_test:", y_test.shape)


x_train= normalize(x_train, axis=1)
x_test= normalize(x_test, axis=1)

y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)


model=Sequential()

model.add(Conv2D(32,(3,3), input_shape=(input_size, input_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test),shuffle= False)
model.save('BrainTumor10EpochsCategorical.h5')
