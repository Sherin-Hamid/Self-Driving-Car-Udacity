---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: ./output_images/undistorted_images/undist2.jpg "Undistorted"
[image2]: ./output_images/binary_images/binary.png "Binary"
[image3]: ./output_images/warped_images/warped.png "Warp Example"
[image4]: ./output_images/final_images/final6.jpg "Output"
[video1]: ./test_videos_output.mp4 "Video"


---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. The `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test images using the `cv2.undistort()` function and obtained, for example, this result: 

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied the distortion correction to the test images using the `cv2.undistort()` function and obtained, for example, this result:

![alt text][image1]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell [2] in the `apply_thresholds()` function in `P2.ipynb`).  Here's an example of my output for this step.

![alt text][image2]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `image_warp()`, which appears in cell [1] in the file `P2.ipynb`. The `image_warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. I chose the source and destination points in the following manner:

```python
src = np.float32([[720,470],[1100,680],[270,720],[580,470]])
dest = np.float32([[1000,0],[1000,680],[180,680],[180,0]])
```

I verified that my perspective transform was working as expected by applying it to the test images and making sure that the warped counterparts have quite parallel lines. Below is the warped image of the test image 'test3.jpg'.

![alt text][image3]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I fit my lane lines with a 2nd order polynomial in the `find_lanes()` and `fit_polynomial()` functions in cell [3] of my code in `P2.ipynb`


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the `radius_curvature()` function in cell [3] of my code in `P2.ipynb`

I did this using the formula from the course material

curvature = (1+(2Ay+B)²)^(3/2) / |2A|
 
fit_x = Ay² + By+ C


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on the image 'test2.jpg':

![alt text][image4]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is the link to the output video

![alt text][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I faced is determining the right points for the src and dest for the perspective transform. I had to try too many values until I got ones that fit the lanes. The pipeline will not succeed in drawing the lane region correctly if these points are not properly determined. Proposing an algorithm that optimizes the selection of these values and sets them automatically will make the pipeline more robust and efficient.

Since the technique depends on images, using unclear images by the pipeline will make it fail. Proposing techniques to assess the quality of the images and fix the issues of the used images a priori will make the pipeline more robust.