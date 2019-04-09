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

[image1]: ./chess.png "Undistorted"
[image2]: ./undist.png "Road Transformed"
[image3]: ./combo.png "Binary Example"
[image4]: ./warp.png "Warp Example"
[image5]: ./lane.png "Fit Visual"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./AdvancedLaneFinder.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

The camera calibration code can be found in the jupyter notebook located in the lesson. I used cv2.findChessboardCorners to store all the cornerpoints for each image where a chess board was found. I next used the cv2.calibrateCamera function to compute the calibration and distortion coefficients. A distortion correction can be seen using the cv2.undistort function in the project file.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
In cells 5-10, I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step. I first calculated the directional gradient in abs_sobel_thresh(), then the magnitude of the gradient using the mag_thresh(), and the direction using dir_threshold() and calculated the color threshold using hls_select(). I used the S channel because it showed very robust results and was also mentioned as being reliable in the lessons. I combined all the threshold and gradients in the combineThresh() function.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in the 12th cell of the IPYNB.  The `corners_unwarp` function takes as inputs an image (`img`). I chose to hardcode the source and destination points globally.  I then used the cv2.getPerspectiveTransform function to tranform the image creating both a src to dst and a dst to src variable. I then warped the perspectie with the cv2.warpPerspective function which returns a warped image.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

First I generated some histogram data to find the base lane lines. I then found every non-zero x and y pixel and stored them. I used the sliding window technique to create a set of windows that essentially slide as the lane changes to find the likely pixels. Essentially, I indentify the window boundaries and the non zero pixels within, and then append them if they are valid pixels. After getting all the good left and right points, I extract the left and right x/y pixel positions and apply a second order polynomial to each. Additionally, I created a second function called search_around_poly() which takes into account sine lines dont move alot from frame to frame so we dont need to do a blind search each time, instead if we just search around the margin in the next frame of video around the line position we can do a targeted search the next frame.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In cell 16 and 17, I calculate curvature and vehicle position. The code is self explanatory:
```
def measure_curvature_real (leftx, rightx, img_shape, xm_per_pix=3.7/800, ym_per_pix = 25/720):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    # generate some fake data
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

     # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)
curvRes = measure_curvature_real(leftx=left_points[0], rightx=right_points[0], img_shape = img.shape)

def carPos(leftx, rightx, img_shape, xm_per_pix=3.7/800):
    # middle of image
    mid_imgx = img_shape[1]//2

    # left/right checker
    balance = (leftx[-1] + rightx[-1])/2

    # horizontal distance
    horiz = (mid_imgx - balance) * xm_per_pix

    return horiz
horiz = carPos(leftx=left_points[0], rightx=right_points[0], img_shape=img.shape)

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
To draw the lines the generated points were mapped back to the image space using the Minv inverse transformation matrix and cv2 poly function. The function was taken from the useful functions part of the project tab.
![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/wp1E4O1AL8s)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I took a very standardized approach basing and learning my code off of the templates provided. I was able to figure out which functions worked and why due to the fact that I was taught them. Without the knowledge of some of these functions and libraries existing the task wouldve been way harder. This project has a lot of flaws still, it is not perfect on lane tracking and it still struggles with light changes. I could improve the project by messing more with the kernels, vertices, and threshold/gradients. I think this would improve the light issue, so it would be interesting to try new combinations.
