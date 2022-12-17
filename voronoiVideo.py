import numpy as np
from scipy.spatial import Voronoi
import cv2 as cv
import os
import time

class VoronoiImage():
    @staticmethod

    def __minMax(points: np.array) -> np.array:
        points = points.astype('float32')
        pointsTranspose = points.transpose()
        maxValueX = pointsTranspose[0].max()
        minValueX = pointsTranspose[0].min()
        maxValueY = pointsTranspose[1].max()
        minValueY = pointsTranspose[1].min()

        for i in range(len(points)):
            points[i][0] = (points[i][0] - minValueX)/(maxValueX - minValueX)
            points[i][1] = (points[i][1] - minValueY)/(maxValueY - minValueY)

        return points


    def checkBluringKernelSizeCorrectness(bluringKernelSize, imageSize):
        allCorrect = True

        if bluringKernelSize == None:
            pass
        elif isinstance(bluringKernelSize, int):
            if bluringKernelSize < 3:
                allCorrect = False
                print("Invorrect bluringKernelSize value. Value must be positive odd int number greater than 1")
            elif bluringKernelSize % 2 == 0:
                allCorrect = False
                print("Invorrect bluringKernelSize value. Value must be odd.")
            elif bluringKernelSize >= min(imageSize):
                allCorrect = False
                print("Incorrect bluringKernelSize value. Value must be smaller than smaller side of image.")
        elif isinstance(bluringKernelSize, str):
            if bluringKernelSize.lower() == 'auto':
                bluringKernelSize = int(min(imageSize) // 200) # 200 is some value regulated kernelSize depends of image shape
                if bluringKernelSize % 2 == 0:
                    bluringKernelSize += 1
            else:
                allCorrect = False
                print("""Incorrect bluringKernelSize data type. Only positive odd int values greater than 1 and smaller than smaller side of image, or None value if you don't wont to use bluring, or 'auto' value if you want to automaticly set the kernelSize.""")
        else:
            allCorrect = False
            print("""Incorrect bluringKernelSize data type. Only positive odd int values greater than 1 and smaller than smaller side of image, or None value if you don't wont to use bluring, or 'auto' value if you want to automaticly set the kernelSize.""")

        return allCorrect, bluringKernelSize


    def generate(image, points, saveInDirectory: bool = False, outputDirectoryPath: str = None, bluringKernelSize = 'auto') -> np.ndarray:
        """
        ### Parameters
        `image`: Array of image. Only list, tuple np.ndarray or string with path to image. Shape of image 
        can be (height, width, channels) or (height, width) in one channel case.

        `points`: Array of points with shape (numberOfPoints, 2). Correct data types is list, tuple or 
        np.ndarry. Or int value with number of random points.

        `saveInDirectory`: Value that decides if you want to save this photo. Only boolean values.

        `outputDirectoryPath`: Path to the folder where you want to save the file. Only string values.

        `bluringKernelSize`: Size of kernel (length of filter side) using to bluring image before generating 
        voronoi diagram. Only positive odd int values greater than 1 and smaller than smaller side of image, 
        or None value if you don't wont to use bluring, or 'auto' value if you want to automaticly set the 
        kernelSize.

        ### Returns
        out: np.ndarray

        Array of voronoi image.
        """

        # ERRORS DETECTION
        allCorrect = True
        fileName = ''
        pointsLen = 1
        kernelSize = 3

        # image
        if isinstance(image, str):
            try:
                fileName = os.path.basename(image).split('.')[0]
                image = cv.imread(image)
            except FileNotFoundError as e:
                allCorrect = False
                print('This file does not exist.')
                print(e)
        elif isinstance(image, np.ndarray):
            if not (len(image.shape) == 2 or len(image.shape) == 3):
                allCorrect = False
                print("""Incorrect image shape. Correct shape is (xSize, ySize, channels) or (xSize, ySize) in one channel case.""")
        elif isinstance(image, list) or isinstance(image, tuple):
            image = np.array(image)
            if not (len(image.shape) == 2 or len(image.shape) == 3):
                allCorrect = False
                print("""Incorrect image shape. Correct shape is (xSize, ySize, channels) or (xSize, ySize) in one channel case.""")
        else:
            allCorrect = False
            print("Incorrect image data type. Only list, tuple, np.ndarray or string with path do image.")

        # points
        if allCorrect:
            if isinstance(points, list) or isinstance(points, tuple) or isinstance(points, np.ndarray):
                if not isinstance(points, np.ndarray):
                    points = np.array(points)

                if not (len(points.shape) == 2 and points.shape[1] == 2):
                    allCorrect = False
                    print("Incorrect points shape. Correct shape is (nPoints, 2). second")
            elif isinstance(points, int):
                if points > 1:
                    points = np.random.random((points, 2))
                else:
                    allCorrect = False
                    print("Incorrect nPoints value. nPoints must be greater or equal than 2.")
            else:
                allCorrect = False
                print("""Incorrect points data type. Only list, tuple or np.array with points coodinations or int value with number of random points.""")

        if allCorrect:
            pointsLen = len(points)

        # saveInDirectory
        if allCorrect:
            if not isinstance(saveInDirectory, bool):
                allCorrect = False
                print("Incorrect saveInDirectory value. Only bool.")

        # outputDirectory
        if allCorrect:
            if outputDirectoryPath is None:
                outputDirectoryPath = ''
            elif isinstance(outputDirectoryPath, str):
                if not os.path.exists(outputDirectoryPath):
                    allCorrect = False
                    print("Incorrect outputDirectoryPath. Path to output directory is wrong or doesn't exist.")
            else:
                allCorrect = False
                print("Incorrect outputDirectoryPath data type. Only string with path to directory.")

        # bluringKernelSize
        if allCorrect:
            allCorrect, bluringKernelSize = VoronoiImage.checkBluringKernelSizeCorrectness(bluringKernelSize, [image.shape[0], image.shape[1]])

        if allCorrect:
            # Params
            image = np.array(image).astype('uint8')
            size = (image.shape[0], image.shape[1])
            nPoints = len(points)

            # Min max scaling and adding outside triangle points
            points = VoronoiImage.__minMax(points)
            points = np.append(points, [[0, 100]], axis=0)
            points = np.append(points, [[-100, -100]], axis=0)
            points = np.append(points, [[100, -100]], axis=0)

            # Generating voronoi mesh
            vor = Voronoi(points)

            # Points and regions validation
            points = vor.points[0:-3]
            pointRegions = vor.point_region[0:-3]
            regions = vor.regions

            reg = []
            ptn = []
            for i in range(len(points)):
                if not -1 in regions[pointRegions[i]]:
                    ptn.append(points[i])
                    reg.append(regions[pointRegions[i]])

            # Points scaling
            ptn = np.array(ptn).transpose()
            ptn[0] = ptn[0] * (size[0] - 1)
            ptn[1] = ptn[1] * (size[1] - 1)
            ptn = ptn.astype('uint16')
            ptn = ptn.transpose()

            # Vertices scaling
            ver = np.array(vor.vertices).transpose()
            ver[0] = ver[0] * (size[0] - 1)
            ver[1] = ver[1] * (size[1] - 1)
            ver = ver.transpose()

            # Blurring
            if not bluringKernelSize == None:
                kernel = np.ones((bluringKernelSize, bluringKernelSize),np.float32) / (bluringKernelSize**2)
                image = cv.filter2D(image, -1, kernel)

            # Colors
            colors = []
            if isinstance(image[0][0], np.ndarray):
                for point in ptn:
                    colors.append((
                        int(image[point[0]][point[1]][0]), 
                        int(image[point[0]][point[1]][1]), 
                        int(image[point[0]][point[1]][2])
                    ))
            elif isinstance(image[0][0], np.uint8):
                for point in ptn:
                    colors.append(int(image[point[0]][point[1]]))

            # Drawing
            outputImage = np.full((size[1], size[0], 3), 255)

            for region, color in zip(reg, colors):
                # seting all polygon tops points
                polygonPoints = []
                for value in region:
                    polygonPoints.append(tuple(ver[value]))

                cv.fillPoly(outputImage, [np.int32(polygonPoints)], color)

            # Image rotating and flipping
            border = int((max(size)-min(size))/2)

            imageTemp = np.zeros((max(size), max(size), 3), dtype='int32')

            if size[0] > size[1]: # vertical
                imageTemp[border:size[1]+border, 0:size[0]] = outputImage
            else: # horizontal
                imageTemp[0:size[1], border:size[0]+border] = outputImage

            outputImage = np.rot90(imageTemp, k=3, axes=(0, 1))

            if size[0] > size[1]: # vertical
                outputImage = outputImage[0:size[0], border:border+size[1]]
            else: # horizontal
                outputImage = outputImage[border:border+size[0], 0:size[1]]

            outputImage = np.fliplr(outputImage)
            outputImage = outputImage.astype('uint8')

            # Image saving
            if saveInDirectory:
                # generating file name
                outputFileName = outputDirectoryPath
                if outputFileName != '':
                    if not (outputFileName[-1] == '/' or outputFileName[-1] == '\\'):
                        outputFileName += '/'
                outputFileName += fileName
                outputFileName += '_' + str(pointsLen) + "nPts"

                # checing if file with outputFileName exist
                iter = -1
                while True:
                    iter += 1
                    if iter == 0:
                        if not os.path.isfile(outputFileName + ".png"):
                            outputFileName += ".png"
                            break
                    else:
                        if not os.path.isfile(outputFileName + "(" + str(iter+1) + ")" + ".png"):
                            outputFileName += "({}).png".format(str(int(iter+1)))
                            break
                
                cv.imwrite(outputFileName, outputImage)

            return cv.cvtColor(outputImage, cv.COLOR_BGR2RGB)


class VoronoiVideo():
    @staticmethod

    def __printEstimatedTime(time: float, withStartTime: bool = False):
        hour   = int(round(time // 3600, 0))
        minute = int(round((time // 60) % 60, 0))
        second = int(round(time % 60, 0))

        if not withStartTime:
            print(" | Estiamted time:", end=' ')

        print(str(hour) + ':', end='')

        if minute < 10:
            print("0" + str(minute) + ':', end='')
        else:
            print(str(minute) + ':', end='')

        if second < 10:
            print("0" + str(second), end='')
        else:
            print(second, end='')


    def generate(videoPath, nPoints: int = 1000, outputDirectoryPath: str = None, outputFrameRate: int = None, frameCounting = True, bluringKernelSize = 'auto'):
        """
        ### Parameters
        `videoPath`: Path to video. Only string values.

        `nPoints`: Number of random points. More points is more calcualtions and longer waiting time. Only 
        int values.

        `outputDirectoryPath`: Directory path where the video will be generated. If outputDirectoryPath is 
        None type then video will be generated in directory where the currently executing file is located. 
        Only string values.

        `outputFrameRate`: Frame per seconds (FPS) in output video. Only int or float values.

        `frameCounting`: Displaying the number of frames in the console. Only bool values.

        `bluringKernelSize`: Size of kernel (length of filter side) using to bluring all of frames before 
        generating voronoi diagram. Only positive odd int values greater than 1 and smaller than smaller 
        side of image, or None value if you don't wont to use bluring, or 'auto' value if you want to 
        automaticly set the kernelSize.

        ### Returns
        out: video in .mp4 data type
        """

        # VALIDATION
        allCorrect = True
        #video
        if isinstance(videoPath, str):
            if not os.path.isfile(videoPath):
                allCorrect = False
                print("Incorrect videoPath. Path is wrong or file doesn't exist.")
        else:
            allCorrect = False
            print("Incorrect videoPath data type. Only string with path to video.")

        # outputDirectory
        if allCorrect:
            if outputDirectoryPath is None:
                outputDirectoryPath = ''
            elif isinstance(outputDirectoryPath, str):
                if not os.path.exists(outputDirectoryPath):
                    allCorrect = False
                    print("Incorrect outputDirectoryPath. Path to output directory is wrong or doesn't exist.")
            else:
                allCorrect = False
                print("Incorrect outputDirectoryPath data type. Only string with path to directory.")

        # nPoints
        if allCorrect:
            if isinstance(nPoints, int):
                if nPoints < 2:
                    allCorrect = False
                    print("Incorrect nPoint value. Only int values greater than 2.")
            else:
                allCorrect = False
                print("Incorrect nPoint value. Only int values greater than 2.")

        # outputFrameRate
        if allCorrect:
            if outputFrameRate is None:
                pass
            elif isinstance(outputFrameRate, int) or isinstance(outputFrameRate, float):
                if outputFrameRate < 1:
                    allCorrect = False
                    print("Incorrect outputFps value. Only positive int or float values.")
            else:
                allCorrect = False
                print("Incorrect outputFps value. Only positive int or float values.")

        # importing video
        if allCorrect:
            capture = cv.VideoCapture(videoPath)

        # bluringKernelSize
        if allCorrect:
            width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
            allCorrect, bluringKernelSize = VoronoiImage.checkBluringKernelSizeCorrectness(bluringKernelSize, [width, height])

        # GENERATING
        if allCorrect:
            listOfFrames = []
            listOfVoronoiFrames = []

            # converter calculeting
            fps = int(round(capture.get(cv.CAP_PROP_FPS), 0))
            if outputFrameRate is None:
                outputFrameRate = fps
            converter = max(fps / outputFrameRate, 1)

            # reading frames
            index = 0
            while True:
                isTrue, frame = capture.read()
                if not isTrue:
                    break
                if index % converter < 1:
                    listOfFrames.append(frame)
                index += 1

            # release memory space
            capture.release()

            # conveting all frames to voronoi
            listLong = len(listOfFrames)
            startTime = time.time()
            startFrameTime = time.time()
            for i, frame in enumerate(listOfFrames):
                if frameCounting:
                    print("Frame: " + str(i+1) + '/' + str(listLong), end=' | ')
                    print(str(round(((i+1) / listLong * 100), 1)) + '%', end='')

                    elapsedTime = time.time() - startFrameTime
                    elapsedTime = elapsedTime * (listLong - (i+1))

                    if i+1 == listLong:
                        print('', end='\r')
                        print(" " * 75, end='\r')
                        print("Done in ", end='')
                        VoronoiVideo.__printEstimatedTime(time.time() - startTime, withStartTime=True)
                        print()
                    else:
                        VoronoiVideo.__printEstimatedTime(elapsedTime)
                        print('', end='\r')
                        
                    startFrameTime = time.time()

                listOfVoronoiFrames.append(VoronoiImage.generate(frame, nPoints, bluringKernelSize=bluringKernelSize))

            # deleting unnecessary list
            del(listOfFrames)
            
            # generating file name
            outputFileName = outputDirectoryPath
            if outputFileName != '':
                if not (outputFileName[-1] == '/' or outputFileName[-1] == '\\'):
                    outputFileName += '/'
            outputFileName += os.path.basename(videoPath).split('.')[0]
            outputFileName += '_' + str(outputFrameRate) + "fps_" + str(nPoints) + "nPts"

            # checing if file with outputFileName exist
            iter = -1
            while True:
                iter += 1
                if iter == 0:
                    if not os.path.isfile(outputFileName + ".mp4"):
                        outputFileName += ".mp4"
                        break
                else:
                    if not os.path.isfile(outputFileName + "(" + str(iter+1) + ")" + ".mp4"):
                        outputFileName += "({}).mp4".format(str(int(iter+1)))
                        break

            # video params
            codec = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
            resolution = (listOfVoronoiFrames[0].shape[1], listOfVoronoiFrames[0].shape[0])

            # exporting video
            videoOutput = cv.VideoWriter(outputFileName, codec, outputFrameRate, resolution)
            for frame in listOfVoronoiFrames:
                videoOutput.write(frame)
            videoOutput.release()
