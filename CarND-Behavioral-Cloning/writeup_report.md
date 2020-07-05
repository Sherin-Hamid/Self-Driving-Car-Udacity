**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./u_data/data/IMG/center_2016_12_01_13_31_14_092.jpg "Center_Camera_Image"
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter size for the first three convolutional layers and 3x3 for the other two. The conv layers are followed by four dense layers. (model.py lines 70-83) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 71). 

#### 2. Attempts to reduce overfitting in the model

Data augmentation has been considered to have more training data and to generalize the model. 

The model was trained and validated on different datasets to ensure that the model was not overfitting (code lines 96&97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very simple architecture, then add layers as needed to improve the model performance.

My first step was to use a convolution neural network model consisting of two layers, convolution and dense.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a bad performance driving the vehicle autonomosly. 

To improve the performance, I modified the model to include more layers. I ended up using the Nvidia model with an extra dense layer at the end with one output.

Then I added a Lambda layer to further improve the performance.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added a Cropping layer to crop out the top and bottom parts of the images that might be confusing to the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 70-83) consisted of a convolution neural network with the following layers:

1- Lambda, input_shape=(160,320,3)
2- Cropping2D: cropping=((70,25), (0,0))
3- Conv2D, with filters=24, kernel_size=5, and activation='relu'
4- Conv2D, with filters=36, kernel_size=5, and activation='relu'
5- Conv2D, with filters=48, kernel_size=5, and activation='relu'
6- Conv2D, with filters=64, kernel_size=3, and activation='relu'
7- Conv2D, with filters=64, kernel_size=3, and activation='relu'
8- Flatten
9- Dense, with output size=100
10- Dense, with output size=50
11- Dense, with output size=10
12- Dense, with output size=1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return back to the center. I also considered using the dataset provided by Udacity for this project. A sample picture can be found below

![alt text][image1]

To augment the dataset, I also flipped images and angles thinking that this would generalize the model and improve the performance. 

After the collection process, I had 38572 number of training data points. I then preprocessed this data by normalizing it using a Lambda layer (lambda x: (x/255.0)-0.5)).

I finally put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by training and performance analysis over different numbers of epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
