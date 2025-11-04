import numpy as np 
import pandas as pd 
from pathlib import Path 
import os.path 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
import tensorflow as tf 
from sklearn.metrics import confusion_matrix, classification_report 
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.preprocessing import image 
image_dir = Path('Food Classification') 
filepaths = list(image_dir.glob(r'**/*.jpg')) 
#using glob to target particular image files 
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths)) 
#separating the class names from the file paths and saving in labels 
filepaths = pd.Series(filepaths, name='Filepath').astype(str) 
labels = pd.Series(labels, name='Label') 
images = pd.concat([filepaths, labels], axis=1) 
category_samples = [] 
for category in images['Label'].unique(): 
category_slice = images.query("Label == @category") 
category_samples.append(category_slice.sample(130, random_state=1)) 
#concatenate category samples 
image_df 
= 
pd.concat(category_samples, 
random_state=1).reset_index(drop=True) 
#sample 100% of the data again after shuffling 
axis=0).sample(frac=1.0,
               image_df['Label'].value_counts() 
train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1) 
#70% training 30% test 
#since we are shuffling, random state = 1 
#limited memory so we train in batches to recycle memory 
train_generator = tf.keras.preprocessing.image.ImageDataGenerator( 
preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input, 
validation_split=0.2 
) 
test_generator = tf.keras.preprocessing.image.ImageDataGenerator( 
preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input 
) 
#this train_images will be passed into our fit function 
train_images = train_generator.flow_from_dataframe( 
dataframe=train_df, 
x_col='Filepath', 
y_col='Label', 
target_size=(224, 224), 
#default image size for mobilenetV2 is 224x224 
color_mode='rgb', 
class_mode='categorical', 
batch_size=32, 
shuffle=True, 
#shuffle after each epoch 
seed=42, 
subset='training' 
#only available if validation_split is used to specify whether to use validation subset 0.2 or 
training subset 
) 
validation_images = train_generator.flow_from_dataframe( 
dataframe=train_df, 
x_col='Filepath', 
y_col='Label', 
target_size=(224, 224), 
color_mode='rgb', 
class_mode='categorical', 
batch_size=32, 
shuffle=True, 
seed=42, 
subset='validation' 
) 
test_images = test_generator.flow_from_dataframe( 
dataframe=test_df, 
x_col='Filepath', 
y_col='Label', 
target_size=(224, 224), 
color_mode='rgb', 
class_mode='categorical', 
batch_size=32, 
shuffle=False 
) 
pretrained_model = tf.keras.applications.MobileNetV2( 
input_shape=(224, 224, 3), 
include_top=False, 
#we dont wanna keep the classification layer of the og dataset on which the model is pretrained 
we just want our dataset's classification layer 
#originally trained on imagenet  dataset 1000 classes 
weights='imagenet', 
#to keep the same weights 
pooling='avg' 
#output is 1d now 
) 
pretrained_model.trainable = False 
#to not change the original imagenet weights 
#We are transfering learning of the model so we keep it as it is 
#This model is good for feature extraction 
#Use the same model, remove the top layer, use your own top layer i.e. classes, dataset 
inputs = pretrained_model.input 
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output) 
#128 neurons 
x = tf.keras.layers.Dense(128, activation='relu')(x) 
outputs = tf.keras.layers.Dense(20, activation='softmax')(x) 
#classification layer 
model = tf.keras.Model(inputs, outputs) 
print(model.summary())
model.compile( 
optimizer='adam', 
loss='categorical_crossentropy', 
#as classes are encoded as vectors by the generator so we use categorical_crossentropy 
metrics=['accuracy'] 
) 
history = model.fit( 
train_images, 
validation_data=validation_images, 
epochs=10, 
#INCREASE NO. OF EPOCHS TO INCREASE ACCURACY (77.46% ACCURACY 
ACHIEVED ON 100 EPOCHS) 
callbacks=[ 
tf.keras.callbacks.EarlyStopping( 
monitor='validation_loss', 
patience=3, 
#when validation loss stops improving for 3 consecutive epochs training will be 
stopped and best epochs weights are restored 
restore_best_weights=True 
) 
] 
) 
results = model.evaluate(test_images, verbose=0) 
#gives loss and accuracy for test set 
print("Test Accuracy: {:.2f}%".format(results[1] * 100)) 
model.save("final_cnn.h5") 
predictions = np.argmax(model.predict(test_images), axis=1) 
cm = confusion_matrix(test_images.labels, predictions) 
clr 
= 
classification_report(test_images.labels, 
target_names=test_images.class_indices, zero_division=0) 
plt.figure(figsize=(6, 6)) 
predictions, 
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False) 
plt.xticks(ticks=np.arange(20) + 0.5, labels=test_images.class_indices, rotation=90) 
plt.yticks(ticks=np.arange(20) + 0.5, labels=test_images.class_indices, rotation=0) 
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
plt.title("Confusion Matrix") 
plt.show() 
print("Classification Report:\n----------------------\n", clr)
