'''
Created on Dec 15, 2020

@author: Ramesh
'''
import pathlib
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab

data_dir = pathlib.Path('/home/test/Documents/DeepLearning/Classification/AOI/Fan/Training/Datasets/Images/Preprocessed')
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Image Count is ',image_count)

# Image parameters
batch_size = 32
img_height = 224
img_width = img_height

# Hyperparameters
learning_rate = 0.0001
tr_epochs = 30

# Creating the training and validation datasets after resizing and data splitting
train_ds = preprocessing.image_dataset_from_directory(
  data_dir,
  labels = 'inferred',
  label_mode = 'categorical',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  color_mode =  "rgb",
  batch_size=batch_size)

val_ds = preprocessing.image_dataset_from_directory(
  data_dir,
  labels = 'inferred',
  label_mode = 'categorical',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  color_mode =  "rgb",
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
# Labels
num_classes = len(class_names)

# Show the shape and labels of one batch in the train_ds dataset
for images,labels in train_ds.take(1):
    print('Batch Shape: ',images.shape)
    print('Labels: ',labels)

# Improve the execution time by prefetching the input data for step 's+1' while the model is executing the step 's'   
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Distributed training with multiple GPUs
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    # load model without classifier layers
    baseModel = VGG16(include_top=False,pooling='avg',weights='imagenet',input_shape=(img_width, img_height, 3),classes=num_classes)
    #baseModel = VGG16(include_top=False,weights='imagenet',input_shape=(img_width, img_height, 3),classes=num_classes)
    # mark loaded layers as not trainable to retain the learned features of VGG16 models
    k = 0
    for layer in baseModel.layers:
        if k < 11:
            layer.trainable = False
        k = k + 1        
    
    # Configuring the FC layers and output
    #flat = Flatten()(baseModel.output)
    # Add a FC NN
    hidden1 = Dense(512, activation='relu')(baseModel.output)
    drop1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(512, activation='relu')(drop1)
    drop2 = Dropout(0.5)(hidden2)
    output = Dense(num_classes, activation='softmax')(drop2)
    # new model
    model = Model(inputs=baseModel.inputs, outputs=output)
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

# Callbacks
# Specify the log directory to the tensorboard
tensorboard = TensorBoard(log_dir="/home/test/workspace/Fan/src/logs/{}".format(datetime.now().strftime("%d-%m-%Y_%H:%M:%S")))
cp_filepath = "/home/test/workspace/Fan/src/checkpoints/best_model_{}.h5".format(datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))
checkpoint = ModelCheckpoint(filepath=cp_filepath, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False, save_freq='epoch', mode='auto')
#early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto')

t1 = datetime.now()
# Train the model
history = model.fit(train_ds,validation_data=val_ds,epochs=tr_epochs,verbose=1,callbacks=[tensorboard,checkpoint])
td = (datetime.now()-t1).total_seconds()/60
print('Execution Time: {} minutes'.format(td))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
pylab.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
pylab.show()

