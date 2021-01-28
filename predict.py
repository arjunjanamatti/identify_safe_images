from __future__ import division, print_function
import glob

# Keras and Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
import os
from glob import glob

# Load the base model and weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

# defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 3 classes - nude, safe and sexy (naming convention followed from dataset)
model.add(Dense(3, activation='softmax'))

# loading the trained weights
model.load_weights("models/weight_80_60000.hdf5")

# compiling the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


def predict_if_safe(images,name = 'nude'):

    results_list = []
    df = pd.DataFrame()
    try:
        prediction_images = []
        for i in range(len(images)):
            img = image.load_img(images[i], target_size=(80, 80, 3))
            img = image.img_to_array(img)
            img = img / 255
            prediction_images.append(img)

        # converting all the images into numpy array
        prediction_images = np.array(prediction_images)
        # extracting features using pre-trained model
        prediction_images = base_model.predict(prediction_images)
        # converting features in one dimensional array
        prediction_images = prediction_images.reshape(prediction_images.shape[0], 2 * 2 * 512)
        # predicting tags for each array
        predictionar = model.predict_classes(prediction_images)

        # creating the tags
        train = pd.read_csv('train_new.csv')
        y = train['class']
        y = pd.get_dummies(y)
        res = y.columns.values[predictionar]
        results_list.append(res)

    except:
        res = "Please upload a valid file"
    df['prediction'] = res
    df['actual'] = np.repeat(np.array([name]), len(res))
    # remove the file after processing
    files = glob('uploads/*')
    # for f in files:
    #     os.remove(f)
    if str(res) == "nude":
        res = "Contains strong adult content."
    elif str(res) == 'sexy':
        res = "Contains adult content. May be inappropriate."
    elif str(res) == 'safe':
        res = "Safe for usage."

    return str(res), df

images = glob("uploads/*")
name = 'nude'
num_images = 6000
a, df = predict_if_safe(images, name)
# df['prediction'] = df['prediction'].map({0:})
df.to_csv('df_{}_{}.csv'.format(name, num_images))
