'''s
Created on Dec 29, 2020

@author: Ramesh
'''
import cv2 as cv
import numpy as np
import pathlib
import tensorflow as tf

class_name = "FAIL"
source_dir = list(pathlib.Path('/home/test/Documents/DeepLearning/Classification/AOI/Fan/Training/Datasets/Images/Raw/{}/'.format(class_name)).glob('*.jpg'))
dest_dir = pathlib.Path('/home/test/Documents/DeepLearning/Classification/AOI/Fan/Training/Datasets/Images/Preprocessed/{}/'.format(class_name))
imagenet_mean = np.array([123.68, 116.779, 103.939], dtype="float32") # ImageNet
print(len(source_dir))

for imgPath in source_dir:    
    img = cv.imread(str(imgPath))        
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)    
    img = tf.image.resize(img,[224,224],antialias=True)
    img = img.numpy()   
    
    # Subtract the ImageNet dataset's per-channel mean from the current dataset 
    for c in range(3):
        img[:,:,c] = img[:,:,c] - imagenet_mean[c]            
    
    cv.imwrite(str(dest_dir/imgPath.name),img)
    '''
    while True:
        cv.imshow('Input',img)
        k = cv.waitKey(1)
        if k%256 == 27 :
            cv.destroyAllWindows()
            break   
    break     
    '''
    
            