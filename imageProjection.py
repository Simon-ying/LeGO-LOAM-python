# There are 4 main jobs in this file
# findStartEndAngle(): find the start and end angle for the lidar
# projectPointCloud(): find ground points and other points that are too far from lidar
# groundRemoval(): remove useless ground points
# cloudSegmentation()

import cv2
import numpy