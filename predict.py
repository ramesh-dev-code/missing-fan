'''
Created on Dec 24, 2020

@author: RAMESH
'''
import cv2 as cv
import tensorflow as tf
import numpy as np
from datetime import datetime


model = tf.keras.models.load_model('/home/test/workspace/Fan/src/checkpoints/best_model_16-03-2021_11:29:20.h5')
train_mean = np.array([123.68, 116.779, 103.939], dtype="float32") # Imagenet
cap_frame = cv.VideoCapture("/dev/video0")
cv.namedWindow("Input Image")
im_name = '/home/test/workspace/Fan/src/test_image.jpg'
res_str = ['FAIL','PASS']

while (cap_frame.isOpened()) :
    success,frame = cap_frame.read()
    if not success:
        print('Frame read failed')
        break
    cv.imshow("Input Image",frame)
    k = cv.waitKey(1)
    # Press Esc to close the input image 
    if k%256 == 27:
        break
    elif k%256 == 32:                           
        #cv.imwrite(im_name,frame)
        print('Before resizing',frame.shape,type(frame))
        t1 = datetime.now()
        # Convert BGR into RGB image
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Resizing the image into 224 x 224 
        img = tf.image.resize(frame,[224,224],antialias=True).numpy()
        print('After resizing: ',img.shape,type(img))        
        # Subracting the per-channel mean from the captured frame
        for c in range(3):
            img[:,:,c] = img[:,:,c] - train_mean[c] 
        cv.imwrite(im_name,img)
        cv.imshow("Preprocessed Image",img)
        k2 = cv.waitKey(0)
        '''
        # Press Enter to close the preprocessed image
        if k2%256 == 13:
            cv.destroyWindow("Preprocessed Image")              
        '''
        img = np.expand_dims(img, axis=0)                        
        print('After data augmentation: ',img.shape,type(img))             
        yhat = model.predict(img)        
        yhat = yhat[0].tolist()
        print(yhat)
        max_ind = yhat.index(max(yhat))        
        print(res_str[max_ind])    
        td = (datetime.now()-t1).total_seconds()
        print('Execution Time: {} seconds'.format(td))             

cap_frame.release()
cv.destroyAllWindows()
'''

# Predicting the label from the saved image
imagenet_mean = np.array([103.939, 116.779, 123.68], dtype="float32")
t1 = datetime.now()
# Read the input image from the path
img = cv.imread('/home/test/Documents/DeepLearning/Classification/AOI/HeatSink/Training/Datasets/Images/PASS/203.jpg')

# Subracting per-channel mean from the captured frame
for c in range(3):
    img[:,:,c] = img[:,:,c] - imagenet_mean[c]

cv.imshow("Test",img)

while True:
    k = cv.waitKey(1)
    if k%256 == 27:
        cv.destroyAllWindows()
        break
# Add a new axis for the batch in the first dimension
img = tf.expand_dims(img, axis=0)
print(img.shape)
# Resize the image to 224 x 224
img = tf.image.resize(img,[224,224],antialias=True)
print('Image shape after resizing: ',img.shape)
model = tf.keras.models.load_model('./checkpoints/best_model_24-12-2020_15:37:17.h5')
yhat = model.predict(img)
print(yhat)
td = (datetime.now()-t1).total_seconds()
print('Execution Time: {} seconds'.format(td))
'''
