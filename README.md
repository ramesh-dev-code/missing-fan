# Image Classification of Missing Fan   
### Objectives   
1. To design the CNN-based deep neural network (DNN) to infer the presence of all fans in the device under test   
2. To develop the image classification application with the trained model to infer the real-time status of fans in the webcam images for automated optical insepction (AOI) applications   

### Fan Images   
**PASS: Presence of all four fans**    
![](https://i.imgur.com/SkbSnmA.png)      
**FAIL: At least one fan is missing**   
![](https://i.imgur.com/JX6PcBQ.png)   

### Dataset and Preprocessing   
1. Dataset: 705 images (PASS), 1000 images (FAIL). The images are captured at different distances from the device and also at different angles to induce more variations in the training (80%) and validation (20%) datasets  
2. Convert the color space of the image from BGR to RGB
3. Resize the image to 224 x 224.    
4. Subtract the per-channel mean of the imagenet dataset (RGB:[123.68, 116.779, 103.939]) from the resized image   
**Preprocessed Images**   
**PASS**   
![](https://i.imgur.com/CJANvVj.png)   
**FAIL**   
![](https://i.imgur.com/YaPcYg3.png)   

### Training
1. Refer to the [Fine-tuning](https://github.com/ramesh-dev-code/misaligned-heat-sink#fine-tuning-steps) steps to train the pre-trained VGG-16 model on the target dataset
2. Unfreeze the last two blocks of conv layers in the VGG network to achieve robust performance on the testing images captured at different angles (not included in the training and validation datasets)   
3. The optimal fully-connected network is identified as two 512-node hidden layers with a 2-node output layer with softmax classifier after several experimental trials
4. Training Epochs: 30, Time: 3 m, Trained Model: best_model_16-03-2021_11:29:20.h5   

## Prediction    
1. Capture the webcam image   
2. Change the color space from BGR into RGB   
3. Resize the image into the dimensions of 224 x 224   
4. Subtract the per-channel mean of the imagenet dataset from the resized image   
5. Predict the output class of the image with the trained model   

```
python3 predict.py
```
### Sample Outputs   
**PASS**   
![](https://i.imgur.com/CkEt6o6.png)   

**FAIL: All fans are missing**   
![](https://i.imgur.com/ARDEvsq.jpg)   
 
**FAIL: 3 fans are missing**   
![](https://i.imgur.com/0fZ5ryZ.png)   

**FAIL: 2 fans are missing**   
![](https://i.imgur.com/z9jNQj4.png)   

**FAIL: 1 fan is missing**   
![](https://i.imgur.com/WU1oRtA.png)   
