## Writeup 

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

[image1]: ./examples/chess_undist.png "Undistorted"
[image2]: ./examples/test3.png "Road Transformed"
[image3]: ./examples/binary_threshold.png "Binary Example"
[image4]: ./examples/perspective_mapping.png "Perspective transform mapping"
[image5]: ./examples/top_view.png "Warp Example"
[image6]: ./examples/fit_lines.png "Fit Visual"
[image7]: ./examples/out_test3.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook "AdvLaneLine.ipynb". I have used the chessboard images provided in the camera_cal folder because as explained in the lectures, the regular pattern in these make them a good candidate for camera calibration.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpts` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpts` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpts` and `imgpts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard image using the `cv2.undistort()` function and obtained this result: 
![alt text][image1]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Now that we have the distortion coefficients and camera matrix in 'dist' and 'mtx', we will use these in the pipeline for both test images and frames in the video to correct for distortion. This is the output of applying distortion correction to one of the test images:

![alt text][image2]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have defined a few functions to apply different kinds of color and gradient thresholds to generate binary images in the 5th block of the "AdvLaneLine.ipynb" notebook. After some trial and error, I ended up using a combination of 'S' channel color (after converting the image from RGB to HLS) and gradient threshold in the 'x' direction to generate a binary image (6th code cell of the Ipython notebook). The choice of the channel to use for color thresholding and the type of gradient thresholding to be used was based on which combination best highlighted the lane lines. Here's an example of my output for this step:

![alt text][image3]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform maps points from an image plane to a different plane of view. To help me visualize how I wanted points from my source image to look like in my destination image (top view), the code in the 7th code cell of the notebook marks source image points by blue dots, and the destination points by red crosses. I have hard-coded both of these such that the src image points are placed with some margin beyond the lane lines (so that they will work for all images taken from the center camera), and the destination points are spaced wide apart. Here's the visualization of these points on the test image:

![alt text][image4]
The code for my perspective transform includes a function called `warpImg()`, which appears in the 8th code cell in the file "AdvLaneLine.ipynb".  The `warpImg()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner (since all the images and video frames are of the same size, these hard-coded values worked for this project):

```python
src = np.float32(
    [[550 - 50, 500],
    [730 + 50, 500],
    [1050 + 50, 680,
    [240 - 50, 680]])
dst = np.float32(
    [[150, 100],
    [img.shape[1] - 150, 100],
    [img.shape[1] - 150, img.shape[0] - 100],
    [150, image.shape[0] - 100]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 500, 500      | 150, 100      | 
| 780, 500      | 1130, 100     |
| 1100, 680     | 1130, 620     |
| 190, 680      | 150, 620      |

Here's a comparison of the images before and after perspective transform. The lines appear more or less parallel in the warped image.

![alt text][image5]
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the block title 'Function to detect and draw lane lines', I have taken the following steps to detect lane lines in the test images and fit a 2nd order polynomial curve to them:
1. As suggested in the lecture videos, the first step was to take a histogram of the binary thresholded image (summing pixels along columns) to find the pixels that were contributing to the lane lines. The histogram shows 2 peaks, indicating each lane line - left and right.
2. The next step is to try and find the midpoints of the left and right lanes. At first, I search for the point with the highest value in the histogram going from the left edge till the midpoint of the histogram. However, in case of images with a lot of shadows, the histogram showed some peaks even in the middle of the lane (where the binary thresholded image showed gradient change). I decided that since the pixels contributing to the left lane line would likely be much more to the left of the midpoint, I would only search for the maximum pixel concentration ('leftMidPoint') from the left of the histogram up to some margin before the midpoint. Similarly for the right lane pixel concentration ('rightMidPoint'), I would start from some margin after the histogram midpoint and search to the end of the histogram.
3. With the previous step, I find the x-position of the base of the left and right lane lines. From this point onward, a sliding window approach is employed to find the pixel positions contributing to the lane lines. A window with some margins is defined around the 'leftMidPoint' and 'rightMidPoint'. The nonzero pixels within this window are recorded as the ones contributing to the left and right lane lines respectively. In the next iteration, the window slides from the previous position depending on which side of the line midpoint showed higher pixel intensity (above a threshold of 100 pixels). Sliding window iterations are run until the entire height of the image is covered.
4. Now that we have the (x,y) coordinates of the pixels contributing to each lane line, I just run the OpenCV polyfit function to finf the coefficients of a 2nd order polynomial to fit these points.

This is how my test image looks after fitting a 2nd order polynomial:

![alt text][image6]
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I computed these in the code cell titled 'Calculate radius of curvature and position wrt center of image' in the notebook. To compute the radius of curvature in real world dimensions, I first found the polyfit coefficients using real world (x,y) coordinates by converting pixels to meters. For the offset, I first computed the offset of the lane center from the image center in pixels, and then converted this difference to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the code cell titled 'Project lane lines back to real world perspective'. For transforming the image back from the top view to the real world view, I use the inverse perspective transformation matrix 'Minv' that I computed by swapping the source and destination images in the warpImg() function. The lane boundaries are identified from the filled green color. Here is an example of my result on a test image:

![alt text][image7]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

For the project video, I have added two more steps to the lane detection pipeline that I used for the test images:
1. I preserve the polyfit 'x' coordinates for the left and right lane lines from the previous frame in objects 'leftLane' and 'rightLane' of class Line. To get the final coordinates contributing the the left and right lines, I take an elementwise average of the coordinates of the newly detected line and the corresponding line from the previous frame. This leads to a more stable transition of lane boundaries from frame to frame.
2. If, even after applying the avergaging mechanism from the previous step, I still happen to get a lane boundaries that are just way off from the true boundaries, I must discard these and use previously found good polyfits. To achieve this, I compute the absolute elementwise difference between line coordinates from the current frame and previous frame for the left and right lines respectively. If at any point, the difference exceeds 25 pixels (chosen by random trial and error), I deem the new line fit to be bad, and use the previous polyfit.

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a hard time filtering the shadows from images. Some techniques I employed to achieve better results were to search for my line centers only up to a certain margin from the histogram midpoint (thus filtering shadows detected in the middle of the lane), and to use an averaging mechanism. However, even these mechanisms fall short in the challenge video, where it appears to have an intersection of a lighter (concrete section) with a darker (asphalt section) patch in the middle of the lane, causing my binary thresholding to detect a line at this boundary. My pipeline clearly fails at such points. I think by using some better preprocessing techniques on the image frame like brightness/contrast manipulations, it might yield better detection. Also, different thresholding techniques could yield better results in different scenarios for instance whether the images are from a bright morning or taken during sunset.

To make the lane detection seem more robust from frame to frame, I could also employ the 'smoothing over N frames' technique described in the tips and tricks section. Instead of blindly searching for the start midpoint of the sliding windows for each frame, I could search around the line coordinates from the previous frame.

The radius of curvature of the left and right lines (maybe their ratio or difference) from a correctly detected frame could serve as a good indicator of whether the lane boundaries detected in another frame are 'good' or not. 
