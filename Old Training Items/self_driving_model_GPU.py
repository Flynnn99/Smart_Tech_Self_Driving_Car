import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPool2D, Dropout, Flatten, Dense
import cv2
import pandas as pd
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa 
import tensorflow as tf 


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def path_leaf(path):
  head,tail = ntpath.split(path)
  return tail

# Training and Validation Split
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    centre,left,right = indexed_data[0], indexed_data[1],indexed_data[2]
    image_path.append(os.path.join(datadir,centre.strip()))
    steering.append(float(indexed_data[3]))
    image_path.append(os.path.join(datadir, left.strip()))
    steering.append(float(indexed_data[3]) + 0.15)
    image_path.append(os.path.join(datadir, right.strip()))
    steering.append(float(indexed_data[3]) - 0.15)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings


#Pre Proceess Images
def preprocess_img(img):
  img = mpimg.imread(img)
  img = img[60:135, :, :]
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3),0)
  img = cv2.resize(img, (200,66))
  img = img/255
  return img

def preprocessed_img_no_imread(img):
  img = img[60:135, :, :]
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3),0)
  img = cv2.resize(img, (200,66))
  img = img/255
  return img

#https://arxiv.org/pdf/1604.07316v1.pdf
def nvidia_model():
    model = tf.keras.Sequential()  # 66x200
    model.add(tf.keras.layers.Conv2D(24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))  # 31 x 98
    model.add(tf.keras.layers.Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))  # 14 x 47
    model.add(tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))  # 5 x 22
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='elu'))  # 3x20
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='elu'))  # 1 x 18
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='elu'))
    model.add(tf.keras.layers.Dense(50, activation='elu'))
    model.add(tf.keras.layers.Dense(10, activation='elu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def zoom(image_to_zoom):
  zoom_func = iaa.Affine(scale=(1,1.3))
  z_image = zoom_func.augment_image(image_to_zoom)
  return z_image

def pan(image_to_pan):
  pan_func = iaa.Affine(translate_percent={"x": (-0.1,0.1), "y": (-0.1,0.1)})
  pan_image = pan_func.augment_image(image_to_pan)
  return pan_image

def img_random_brightness(image_to_brighten):
  bright_func = iaa.Multiply((0.2,1.2))
  bright_image = bright_func.augment_image(image_to_brighten).astype("uint8")
  return bright_image

def img_random_flip(image_to_flip,steering_angle):
  flipped_image = cv2.flip(image_to_flip,1)
  steering_angle = -steering_angle
  return flipped_image, steering_angle

def random_augment(image_to_augment,steering_angle):
  augment_image = mpimg.imread(image_to_augment)
  if np.random.rand() < 0.5:
    augment_image = zoom(augment_image)
  if np.random.rand() < 0.5:
    augment_image = pan(augment_image)
  if np.random.rand() < 0.5:
    augment_image = img_random_brightness(augment_image)
  if np.random.rand() < 0.5:
    augment_image, steering_angle = img_random_flip(augment_image,steering_angle)
  return augment_image, steering_angle

def batch_generator(image_paths, steering_ang, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            if is_training:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = tf.keras.preprocessing.image.load_img(image_paths[random_index])
                im = tf.keras.preprocessing.image.img_to_array(im)
                steering = steering_ang[random_index]

            im = preprocessed_img_no_imread(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steering)


datadir = "C:\\Users\\kylem\\Desktop\\CA2-SelfDriving-main\\TrackOne"
columns = ["center", "left","right","steering","throttle","reverse","speed"]
data = pd.read_csv(os.path.join(datadir,"driving_log.csv"),names=columns)
pd.set_option('display.max_columns', 7)
print(data.head)

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

num_bins = 25
samples_per_bin = 400
hist,bins = np.histogram(data['steering'],num_bins)
centre = (bins[:-1] + bins[1:]) * 0.5
# plt.bar(centre, hist, width=0.05)
# plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
# plt.show()

remove_list = []
print('total data:', len(data))
for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)
print('removed', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))

# hist,bins = np.histogram(data['steering'],num_bins)
# plt.bar(centre, hist, width=0.05)
# plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
# plt.show()

image_paths, steerings = load_img_steering(datadir + '\\IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

# fig, axes = plt.subplots(1,2, figsize=(12,4))
# axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
# axes[0].set_title('Training Set')
# axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
# axes[1].set_title('Validation Set')
# plt.show()

# image = image_paths[100]
# original_img = mpimg.imread(image)
# preprocessed_img = preprocess_img(image)
# fig, axes = plt.subplots(1,2, figsize=(15,10))
# fig.tight_layout()
# axes[0].imshow(original_img)
# axes[0].set_title('Original Image')
# axes[1].imshow(preprocessed_img)
# axes[1].set_title('Preprocessed Image')
# plt.show()

# image = image_paths[random.randint(0,1000)]
# original_img = mpimg.imread(image)
# zoomed_image = zoom(original_img)
# fig, axes = plt.subplots(1,2, figsize=(15,10))
# fig.tight_layout()
# axes[0].imshow(original_img)
# axes[0].set_title('Original Image')
# axes[1].imshow(zoomed_image)
# axes[1].set_title('Zoomed Image')
# plt.show()

# image = image_paths[random.randint(0,1000)]
# original_img = mpimg.imread(image)
# panned_image = pan(original_img)
# fig, axes = plt.subplots(1,2, figsize=(15,10))
# fig.tight_layout()
# axes[0].imshow(original_img)
# axes[0].set_title('Original Image')
# axes[1].imshow(panned_image)
# axes[1].set_title('Panned Image')
# plt.show()

# image = image_paths[random.randint(0,1000)]
# original_img = mpimg.imread(image)
# brightened_image = img_random_brightness(original_img)
# fig, axes = plt.subplots(1,2, figsize=(15,10))
# fig.tight_layout()
# axes[0].imshow(original_img)
# axes[0].set_title('Original Image')
# axes[1].imshow(brightened_image)
# axes[1].set_title('Brightened Image')
# plt.show()

# random_index = random.randint(0,1000)
# image = image_paths[random_index]
# steering = steerings[random_index]
# original_img = mpimg.imread(image)
# flipped_image, flipped_steering = img_random_flip(original_img,steering)
# fig, axes = plt.subplots(1,2, figsize=(15,10))
# fig.tight_layout()
# axes[0].imshow(original_img)
# axes[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering))
# axes[1].imshow(flipped_image)
# axes[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering))
# plt.show()

# ncols = 2
# nrows = 10
# fig, axes = plt.subplots(nrows, ncols, figsize=(15,50))
# fig.tight_layout()
# for i in range(10):
#   randnum = random.randint(0,len(image_paths)-1)
#   random_image = image_paths[randnum]
#   random_steering = steerings[randnum]
#   original_img = mpimg.imread(random_image)
#   augmented_image, steering = random_augment(random_image,random_steering)
#   axes[i][0].imshow(original_img)
#   axes[i][0].set_title('Original Image')
#   axes[i][1].imshow(augmented_image)
#   axes[i][1].set_title('Augmented Image')
# plt.show()

model = nvidia_model()
print(model.summary())

history = model.fit(batch_generator(X_train, y_train, 200, 1),steps_per_epoch=100,epochs=30,validation_data=batch_generator(X_valid, y_valid, 200, 0),validation_steps=200,verbose=1,shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('model.h5')

