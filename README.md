# Voronoi Video
> Module to generating voronoi videos and images.

> Project [_here_](https://github.com/Skamlo/VoronoiVideo)


## General Information and Setup

#### Setup
If you want to run this module you have to install three packages - `numpy`, `cv2` and `scipy`. You can do it with pip method by typing `pip install numpy cv2 scipy` cammand in system console.

#### Generating Voronoi Video
from voronoiImage import VoronoiImage

and another set parameters:

`image`: Array of image. Only list, tuple np.ndarray or string with path to image. Shape of image can be (height, width, channels) or (height, width) in one channel case.

`points`: Array of points with shape (numberOfPoints, 2). Correct data types is list, tuple or np.ndarry. Or int value with number of random points.


#### Generating Voronoi Image
from voronoiImage import VoronoiVideo

and another set parameters:

`videoPath`: Path to video. Only string values.

`nPoints`: Number of random points. More points is more calcualtions and longer waiting time. Only int values.

`outputDirectoryPath`: Directory path where the video will be generated. If outputDirectoryPath is None type then video will be generated in directory where the currently executing file is located. Only string values.

`outputFrameRate`: Frame per seconds (FPS) in output video. Only int or float values.

`frameCounting`: Displaying the number of frames in the console. Only bool values.


#### Results
In result, you get these types of images:

<!-- image source: https://commons.wikimedia.org/wiki/File:Girl_with_a_Pearl_Earring.jpg -->
![normal](img/The_Girl_With_The_Pearl_Earring.png.png)
![voronoi](img/The_Girl_With_The_Pearl_Earring_5000nPts.png.png)


## Technologies Used
- numpy - version 1.23.4
- cv2 - version 22.3.1
- scipy - version 1.8.0


## Contact
email: maksymiliannorkiewicz@gmail.com
