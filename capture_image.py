'''
Created on Dec 15, 2020

@author: Ramesh
'''
import cv2 as cv
cap_frame = cv.VideoCapture("/dev/video0")
cv.namedWindow("Fan")
count = 1

while (cap_frame.isOpened()) :
    success,frame = cap_frame.read()
    if not success:
        print('Frame read failed')
        break
    cv.imshow("Fan",frame)
    k = cv.waitKey(1) 
    if k%256 == 27:
        break
    elif k%256 == 32:
        im_name = '/home/test/Documents/DeepLearning/Classification/AOI/Fan/Training/Datasets/Images/Raw/FAIL/{}.jpg'.format(count)              
        cv.imwrite(im_name,frame)        
        count = count + 1

cap_frame.release()
cv.destroyAllWindows()