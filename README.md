# Voronoi Video
> Module to generating voronoi videos and images.

> Project [_here_](https://github.com/Skamlo/VoronoiVideo)


## General Information and Setup

### Setup
If you want to run this module you have to install three packages - `numpy`, `cv2` and `scipy`. You can do it with pip method by typing `pip install numpy cv2 scipy` cammand in system console.


## Generating Voronoi Image
<span style="color:#9579c0">from</span> <span style="color:#4ec9b0">voronoiVideo</span> <span style="color:#9579c0">import</span> <span style="color:#4ec9b0">VoronoiImage</span>

and another set parameters:

`image`: Array of image. Only list, tuple np.ndarray or string with path to image. Shape of image can be (height, width, channels) or (height, width) in one channel case.

`points`: Array of points with shape (numberOfPoints, 2). Correct data types is list, tuple or np.ndarry. Or int value with number of random points.

`saveInDirectory`: Value that decides if you want to save this photo. Only boolean values.

`outputDirectoryPath`: Path to the folder where you want to save the file. Only string values.

`bluringKernelSize`: Size of kernel (length of filter side) using to bluring image before generating voronoi diagram. Only positive odd int values greater than 1 and smaller than smaller side of image, or None value if you don't wont to use bluring, or 'auto' value if you want to automaticly set the kernelSize.

`numberOfLloydsIters`: Number of iterations in Lloyds Algorythm. Recommended value is max(20, Npoints/50), but if you set numberOfLloydsIters to "auto" then this value will set automatically. Only int values.


### Results

<!-- image source: https://commons.wikimedia.org/wiki/File:Girl_with_a_Pearl_Earring.jpg -->
<p float="left">
  <img src="img/The_Girl_With_The_Pearl_Earring.png" width="32%" />
  <img src="img/The_Girl_With_The_Pearl_Earring_5000nPts.png" width="32%" />
  <img src="img/The_Girl_With_The_Pearl_Earring_Lloyds_Algorythm_5000nPts.png" width="32%" />
</p>


## Generating Voronoi Video
<span style="color:#9579c0">from</span> <span style="color:#4ec9b0">voronoiVideo</span> <span style="color:#9579c0">import</span> <span style="color:#4ec9b0">VoronoiVideo</span>

and another set parameters:

`videoPath`: Path to video. Only string values.

`nPoints`: Number of random points. More points is more calcualtions and longer waiting time. Only int values.

`outputDirectoryPath`: Directory path where the video will be generated. If outputDirectoryPath is None type then video will be generated in directory where the currently executing file is located. Only string values.

`outputFrameRate`: Frame per seconds (FPS) in output video. Only int or float values.

`frameCounting`: Displaying the number of frames in the console. Only bool values.

`bluringKernelSize`: Size of kernel (length of filter side) using to bluring all of frames before generating voronoi diagram. Only positive odd int values greater than 1 and smaller than smaller side of image, or None value if you don't wont to use bluring, or 'auto' value if you want to automaticly set the kernelSize.

`numberOfLloydsIters`: Number of iterations in Lloyds Algorythm. Recommended value is max(20, Npoints/50), but if you set numberOfLloydsIters to "auto" then this value will set automatically. Only int values.


### Results

<p float="left">
  <img src="img/blade_runner_Lloyd.gif" width="98%" />
</p>


## Technologies Used
- numpy - version 1.23.4
- cv2 - version 22.3.1
- scipy - version 1.8.0


## Contact
email: maksymiliannorkiewicz@gmail.com
