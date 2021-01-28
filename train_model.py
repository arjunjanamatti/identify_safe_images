try:
    import cv2  # for capturing videos
    import math  # for mathematical operations
    import pandas as pd
    import numpy as np  # for mathematical operations
    import tensorflow.keras.utils as np_utils
    from skimage.transform import resize  # for resizing images
    from sklearn.model_selection import train_test_split
    from glob import glob
    from tqdm import tqdm
    import os
    # tensorflow imports
    import tensorflow as tf
    from keras_preprocessing import image
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
    from tensorflow.keras.preprocessing import image

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    print('All Modules loaded successfully')
except Exception as e:
    print('Module {} had as issue'.format(e))


main_directory = 'C:/Users/Arjun Janamatti/Documents/image_classification/nude_sexy_safe_v1_x320/training/'
sub_dir = os.listdir(main_directory)

# for augmented images
dict_1 = {}
for sub in sub_dir:
    if 'aug' in sub:
        dict_1[sub] = len(os.listdir(main_directory+sub))
        print(sub, len(os.listdir(main_directory+sub)))

# creating the base model of pre-trained VGG16 model
#base_model = VGG16(weights='imagenet', include_top=False,input_shape=(100,100,3))
base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(80,80,3))

images_list = []
for sub in dict_1:
    images_list.append(os.listdir(main_directory+sub))

new_images_list = []
name_index = list(dict_1.keys())
flat_image_name = []
flat_image_label = []
for index, i in enumerate(images_list):
    for indes_i, j in enumerate(i):
        if indes_i < 30000:
            flat_image_name.append(name_index[index]+'/'+j)
            flat_image_label.append(name_index[index])

# creating an empty list
train_image = []
# for loop to read and store frames
pd_image_name = []
pd_image_label = []

for i in tqdm(range(len(flat_image_name))):
    try:
        # loading the image and keeping the target size as (224,224,3)
        img = image.load_img(main_directory + flat_image_name[i], target_size=(80, 80, 3))

        # converting it to array
        img = image.img_to_array(img)

        # normalizing the pixel value
        img = img / 255

        # appending the image to the train_image list
        train_image.append(img)

        pd_image_name.append(flat_image_name[i])
        pd_image_label.append(flat_image_label[i])

    except:
        pass

# converting the list to numpy array
X = np.array(train_image)

train_data = pd.DataFrame()
train_image_name = []
train_class_label = []
# for sub in dict_1:
#     trial_list = []
#     trial_list.append(os.listdir(main_directory+sub))
#     print(len(trial_list),trial_list[0][0])
#     for index in range(dict_1[sub]):
#         train_image_name.append(sub+'/'+trial_list[0][index])
#         train_class_label.append(sub)
train_data['image'] = pd_image_name
train_data['class'] = pd_image_label
train_data.to_csv('train_new_1.csv',header=True, index=False)
print('Shape of the dataframe: ', train_data.shape)
train_data.head()

train_data['class'] = train_data['class'].map({'nude_aug': 'nude',
                                               'safe_aug': 'safe',
                                               'sexy_aug': 'sexy'})

# separating the target
y = train_data['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)
print('X_train creating the training and validation set completed.')

# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print('X_train creating dummies of target variable for train and validation set completed.')

# extracting features for training frames
X_train = base_model.predict(X_train)
print('X_train extracting features completed.')
# extracting features for validation frames
X_test = base_model.predict(X_test)

# reshaping the training as well as validation frames in single dimension
X_train = X_train.reshape(X_train.shape[0], 2*2*512)
print('X_train reshaping completed.')
X_test = X_test.reshape(X_test.shape[0], 2*2*512)

# normalizing the pixel values
max = X_train.max()
print(max)
X_train = X_train/max
X_test = X_test/max

#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weight_80_90000.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

y_train = y_train.to_numpy(copy=False)
y_test = y_test.to_numpy(copy=False)

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# training the model
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)