
# coding: utf-8

# In[1]:

import csv
import opencv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, Flatten, Convolution2D, Lambda
from keras.optimizers import Adam
from keras.backend import tf as ktf
from keras.layers.core import Lambda
import cv2
import os
import matplotlib.image as mpimg
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image


# In[3]:

def getData(path_data, flag = False):
    with open(path_data + '/driving_log.csv') as csvfile:
        cols = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']
        data = pd.read_csv(csvfile, names=cols, header=1)
    return data
  
# Training images contained in 3 folders "data" - Udacity, "additional_data" = collected for 3 laps,
# "reverse_data" - data for driving in opposite direction - 3 laps.
data_folders = ['data', 'additional_data', 'reverse_data']
i = 0
data = [0, 0, 0]

for folder in data_folders:
    path_data = folder
    data[i] = getData(path_data)
    i = i + 1
    
frames = [data[0], data[1], data[2]]
result = pd.concat(frames)
result = result[result["Steering Angle"] != 0]

# Removing 65% data with steering angle 0
remove, keep = train_test_split(result, test_size = 0.35)

final_df = [keep, result]
final_df = pd.concat(final_df)


# In[4]:

images = final_df[['Center Image', 'Left Image', 'Right Image']]
angles = final_df['Steering Angle']

# Using train_test_split to split 15% data for testing
train_images, validation_images, train_angles, validation_angles = train_test_split(images, angles, test_size=0.15, random_state=21)


# In[8]:

# Modified nvidia model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(80, 320, 3)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(1))

# In[9]:

def get_image(path, flip=False):
    image = Image.open(path.strip())
    
    # flip
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = np.array(image, np.float32)
    # Crop image
    image = image[50:130, :]
    
    return image


# In[6]:



def generator(images, angles, batch_size = 64,  augment=True):
    batch_img = []
    batch_ang = []
    sample_idx = 0  
    idxs = np.arange(len(images))
    
    while True:
        np.random.shuffle(idxs)
        
        for i in idxs:
            sample_idx = sample_idx + 1
            
            # Center image & steering angle
            batch_img.append(get_image((images.iloc[i]['Center Image'])))
            batch_ang.append(float(angles.iloc[i]))
            
            if augment:
                
                # Left image & adjust steering angle
                batch_img.append(get_image((images.iloc[i]['Left Image'])))
                batch_ang.append(min(1.0, float(angles.iloc[i]) + 0.25))

                # Right image & adjust steering angle
                batch_img.append(get_image((images.iloc[i]['Right Image'])))
                batch_ang.append(max(-1.0, float(angles.iloc[i]) - 0.25))
                
                # Flip image & invert angle
                batch_img.append(get_image((images.iloc[i]['Center Image']), True))
                batch_ang.append((-1.) * float(angles.iloc[i]))
                
            if (sample_idx % len(images)) == 0 or (sample_idx % batch_size) == 0:
                yield np.array(batch_img), np.array(batch_ang)
                batch_img = []
                batch_ang = []


# In[ ]:

nb_epoch = 20
lr = 0.0001

generator_train = generator(train_images, train_angles)
generator_validation = generator(validation_images, validation_angles, augment=False)

model.compile(loss='mse', optimizer=Adam(lr))
history = model.fit_generator(generator_train, samples_per_epoch=4*len(train_images), nb_epoch=nb_epoch,
    validation_data=generator_validation, nb_val_samples=len(validation_images))


# In[ ]:

print("Save Model")
model.save('model.h5', True)
print("Model Saved")
# In[ ]:



