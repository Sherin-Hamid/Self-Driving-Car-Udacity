# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consists of 5 steps:

1) Convert the images to grayscale. Define the kernel size to (5,5) and apply Gaussian smoothing.
2) Define the parameters for Canny and apply it to the blurred gray image from step 1. Considering the recommendation for the 1:2 or 1:3 ratio of thresholds, I used low_threshold = 50 and high_threshold = 150.
3) Create a masked edges image of the output of step 2.
4) Define the Hough transform parameters and make a blank the same size as the original image to draw on. Apply the Hough transformation to the masked image.
5) Create a "color" binary image to combine with line image. Draw the lines on the edge image from step 4.

I then modified the draw_lines function to draw full lane lines, instead of the intermittent ones. I did so by looping on the lines and classifying them as left or right lines. On the lane lines of each lane, I applied a polyfit function to fit them in a line. I used the coefficients of the resulting poly function to calculate the corresponding x values and then used these values for plotting a corresponding solid line.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lane lines are not well distinctive due to weather conditions affecting the quality of the captured images/videos. 

Another shortcoming could be the limitation of the current pipeline to only detection of straight lane lines. What about when we have to detect lines on curved roads?


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to improve the saturation of the images before applying the pipeline.

Another potential improvement could be to add a function that splits the curved lane lines into a series of connected straight lines. This will help apply the current pipeline to any shape of lane lines.
