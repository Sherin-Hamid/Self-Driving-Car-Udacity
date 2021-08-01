# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dataset_exploration.png "Visualization"
[image2]: ./test_1.jpg "Traffic Sign 1"
[image3]: ./test_2.jpg "Traffic Sign 2"
[image4]: ./test_3.jpg "Traffic Sign 3"
[image5]: ./test_4.jpg "Traffic Sign 4"
[image6]: ./test_5.jpg "Traffic Sign 5"
[image7]: ./test_6.jpg "Traffic Sign 6"
[image8]: ./test_7.jpg "Traffic Sign 7"


---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. That was followed by normailzing the images. However, the results obtained were not promising. I thought of testing the model on the original images without preprocessing. The results were so much better, so I decided to ignore the preprocessing steps and stick to the original images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding 					|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, VALID padding 					|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, VALID padding 					|
| RELU					|												|
| Max pooling			| 2x2 stride 									|
| Dropout				| 												|
| Flatten		 	 	| 	 	 	 									|
| Fully connected		| 												|
| RELU					|												|
| Dropout				| 												|
| Fully connected		| 												|
| RELU					|												|
| Fully connected		| 												|
| RELU					|												|
| Fully connected		| 												|
| Softmax				| Outpu = 43 (number of sign categories)		|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer with BATCH_SIZE = 128, EPOCHS = 60, rate = 0.001, mu = 0, sigma = 0.1

To decide on the parameters above, I tried different values of them and picked the one with the best results. For example, for thr number of epochs, I tried 40, 50, 60, and 100. For the batch size, I tried 64 but the results were not good and the training with slower than 128. I tried 256 - it was faster but the results were a bit worse. For the training rate, I tried 0.01, 0.001, and 0.0001. The results of 0.01 and 0.0001 were worse than 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.969 
* test set accuracy of 0.961

I first tried a model with a max_pool layer after each convolutional layer, but it didn't give good results. I then tried removing them and keeping it only after the last convolutional layer and it gave better results.
I also had another model with a dropout layer after each fully-connected one, but the results were not that good. Results improved after I removed these layers and had only one after the first fully-connected layer, as in the final model above.

I also tried training the model first with 20 epochs, then realized that increasing the number of epochs can significantly improve the accuracy. So, I increased the number of epochs to 50.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. 

Here are seven German traffic signs that I found on the web, along with some challenges the model may encounter in predicting their classes

      Image                           Challenge

![alt text][image2  =100x20]  the sign is not centered in the image
![alt text][image3  =100x20]  the contrast between the sign color and the background might not be strong enough to distinguish the sign 
![alt text][image4  =100x20]  there are other objects in the image (clouds and trees)
![alt text][image5  =100x20]  there is text in this image which might be confusing for the model. Also, there is a building on the right                        of the image with colors close to the sign's
![alt text][image6  =100x20]  the resolution of the top of the image is not perfect
![alt text][image7  =100x20]  I don't see tricky issues with this image 
![alt text][image8  =100x20]  the children area sign is close to the pedestrian crossing sign, which might be confusing for the model



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road   		| Priority road 								| 
| Children crossing 	| Children crossing 							|
| Turn right ahead 		| Yield 										|
| Right-of-way at the   | Right-of-way at the next intersection 		
  next intersection	    | 					 							|
| Double curve 			| Priority road 	 							|
| Stop 					| Stop      									|
| Road work 			| Road work      								|


The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 0.714. This compares moderately to the accuracy on the test set of 0.961

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top five softmax probabilities for the images are as below, which show that the model is significantly certain of the predictions

For test image 1: the model's softmax probabilities are [ 1.  0.  0.  0.  0.] for class labels [12 40  7  1 13] respectively.
For test image 2: the model's softmax probabilities are [ 1.  0.  0.  0.  0.] for class labels [28  2 24  1 25] respectively.
For test image 3: the model's softmax probabilities are [ 0.93000001  0.02  0.01   0.01   0. ] for class labels [33 35 11  5 34] respectively.
For test image 4: the model's softmax probabilities are [ 1.  0.  0.  0.  0.] for class labels [11 27 30 21 24] respectively.
For test image 5: the model's softmax probabilities are [ 0.25999999  0.2   0.18000001  0.1   0.06 ] for class labels [12 18 39 37  1] respectively.
For test image 6: the model's softmax probabilities are [ 1.  0.  0.  0.  0.] for class labels [14  1  4  2  5] respectively.
For test image 7: the model's softmax probabilities are [ 0.99000001  0.   0.   0.    0. ] for class labels [25 26 18 11 29] respectively.


