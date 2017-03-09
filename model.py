import csv
from random import shuffle
import cv2
import numpy as np
import sklearn
import random
import os

samples = []
sampledrop_prob = 6
sampledrop_minimum = .2
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if abs(float(line[3])) > sampledrop_minimum:
            samples.append(line)
        elif random.randint(1, 10) > sampledrop_prob:
            samples.append(line)
with open('./old_driving_log.csv') as csvfile:
    reversereader = csv.reader(csvfile)
    for line in reversereader:
        if abs(float(line[3])) > sampledrop_minimum:
            samples.append(line)
        elif random.randint(1, 10) > sampledrop_prob:
            samples.append(line)
with open('./curves_driving_log.csv') as csvfile:
    reversereader = csv.reader(csvfile)
    for line in reversereader:
        if abs(float(line[3])) > sampledrop_minimum:
            samples.append(line)
        elif random.randint(1, 10) > sampledrop_prob:
            samples.append(line)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))
def preprocessgen(path, angle, rightcenleft):
    #-1 for right, 0 for center, 1 for left
    correction = 0.1
    name = path.split('/')[-1]
    image = np.array(cv2.cvtColor((cv2.imread(name)), cv2.COLOR_BGR2RGB))
    image = cv2.resize(image,(80, 40), interpolation=cv2.INTER_AREA)
    if angle < 0 and rightcenleft > 0:
        angle = angle + correction
    elif angle < 0 and rightcenleft < 0:
        angle = angle - ((angle * angle)/.43) - correction
    elif angle > 0 and rightcenleft > 0:
        angle = angle + ((angle * angle)/.43) + correction
    elif angle > 0 and rightcenleft < 0:
        angle = angle - correction
    else:
        angle = angle + (correction * rightcenleft)
    if random.randint(1, 10) >= 6:
        image = np.fliplr(image)
        angle = -angle
    return image, angle

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                center_image, center_angle = preprocessgen(batch_sample[0], float(batch_sample[3]), 0)
                left_image, left_angle = preprocessgen(batch_sample[1], float(batch_sample[3]), 1)
                right_image, right_angle = preprocessgen(batch_sample[2], float(batch_sample[3]), -1)
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((10,5), (0,0)), input_shape = (40, 80, 3)))
model.add(Lambda(lambda x: (x/255) - 0.5))
model.add(Convolution2D(24, 5, 5,
                        border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5,
                       border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(48, 3, 3,
                       border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,
                       border_mode='valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=
            (len(train_samples)*3), validation_data=validation_generator,
            nb_val_samples=(len(validation_samples)*3), nb_epoch=4, verbose=2)
model.save('model.h5')

print(history_object.history.keys())

#plot
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
