# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "Model Visualization"
[image2]: ./examples/center_line.jpg "Grayscaling"
[image3]: ./examples/center_line.jpg "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the stabdard NVIDIA structure from https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	cropping2d_1 (Cropping2D)    (None, 45, 160, 3)        0         
	_________________________________________________________________
	lambda_1 (Lambda)            (None, 45, 160, 3)        0         
	_________________________________________________________________
	Conv_stage_1 (Conv2D)        (None, 23, 80, 24)        1824      
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 22, 79, 24)        0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 11, 40, 36)        21636     
	_________________________________________________________________
	max_pooling2d_2 (MaxPooling2 (None, 10, 39, 36)        0         
	_________________________________________________________________
	conv2d_2 (Conv2D)            (None, 5, 20, 48)         43248     
	_________________________________________________________________
	max_pooling2d_3 (MaxPooling2 (None, 4, 19, 48)         0         
	_________________________________________________________________
	conv2d_3 (Conv2D)            (None, 4, 19, 64)         27712     
	_________________________________________________________________
	max_pooling2d_4 (MaxPooling2 (None, 3, 18, 64)         0         
	_________________________________________________________________
	conv2d_4 (Conv2D)            (None, 3, 18, 64)         36928     
	_________________________________________________________________
	max_pooling2d_5 (MaxPooling2 (None, 2, 17, 64)         0         
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 2176)              0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 1164)              2534028   
	_________________________________________________________________
	dense_2 (Dense)              (None, 100)               116500    
	_________________________________________________________________
	dense_3 (Dense)              (None, 50)                5050      
	_________________________________________________________________
	dense_4 (Dense)              (None, 10)                510       
	_________________________________________________________________
	dense_5 (Dense)              (None, 1)                 11        
	=================================================================
	Total params: 2,787,447
	Trainable params: 2,787,447
	Non-trainable params: 0
	_________________________________________________________________

#### 2. Attempts to reduce overfitting in the model

I didn't use droputs - I thought I would need to, but the final result was pretty good without that. I was only randomizing the train data


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 67).

#### 4. Appropriate training data

I've used the detaul training data from the Udacity repo. For each row I had been using the center image, flipped image, left and right image. The correction factor for left and right images is 0.15 ( line 14 ) 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I've used the NVIDIA model architecture which works pretty good.


#### 2. Final Model Architecture

The final model architecture (model.py lines 26-68) 

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I'd been using the sample data from the Udacity repository, which in my opinion is pretty good for training for the track number 1.

![alt text][image2]


