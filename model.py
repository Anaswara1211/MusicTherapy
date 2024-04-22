import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Define data directory and classes
Datadirectory = "fer/fer2013/train"
Classes = ["Angry", "Happy", "Neutral", "Sad"]

# Display one image from each category
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break

# Image preprocessing parameters
img_size = 64

# Read and preprocess training images
training_Data = []
def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_Data()
random.shuffle(training_Data)

# Split data into training and validation sets
X = []
y = []
for features, label in training_Data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X / 255.0
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Create new model architecture
inputs = keras.Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(Classes), activation='softmax')(x)

new_model = keras.Model(inputs, outputs)
# Compile and train the model
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = new_model.fit(X_train, y_train, epochs=60, validation_data=(X_val, y_val))


model_json = new_model.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)

new_model.save_weights("classifier_weights.weights.h5")
print("Saved model to disk")