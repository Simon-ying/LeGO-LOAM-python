# There are 4 main jobs in this file
# findStartEndAngle(): find the start and end angle for the lidar
# projectPointCloud(): find ground points and other points that are too far from lidar
# groundRemoval(): remove useless ground points
# cloudSegmentation(): Use method in Fast Range Image-based Segmentation of Sparse 3D Laser Scans for Online Operation

# Default parameter:
# upper-fov = 15 degrees
# lower-fov = -25 degrees
# N_SCAN = channels = 64
# range = 100 m
# points-per-second = 500000
# rotation_frequency = 10 Hz
# groundScanInd = 15 for Ouster OS1-64

# ang_res_x = 360*rotation_frequency*channels / points_per-second = 0.2304
# ang_res_y = (upper-fov - lower-fov) / (channels -1) = 40 / 63 = 0.635
# Horizon_SCAN = points_per-second / (rotation_frequency*channels) = 1562

'''
// Ouster OS1-64
// extern const int N_SCAN = 64;
// extern const int Horizon_SCAN = 1024;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 15;
'''

from dataHandler import loadLidarData
from dataHandler import visulizeLiadarData

import numpy as np
import math
from queue import Queue

labelCount = 1
upper_fov = 15
lower_fov = -25
N_SCAN = 64
points_per_second = 500000
rotation_frequency = 10
groundScanInd = 15

Horizon_SCAN = round(points_per_second / (rotation_frequency * N_SCAN))
ang_res_x = 360 / Horizon_SCAN
ang_res_y = (upper_fov - lower_fov) / (N_SCAN - 1)


def rad2deg(theta):
    return theta * 180 / math.pi

def findStartEndAngle(lidarData):
    pointsSize = lidarData.shape[0]
    startAngle = -math.atan2(lidarData[0, 1], lidarData[0, 0])
    endAngle = -math.atan2(lidarData[pointsSize-1, 1], lidarData[pointsSize-1, 0]) + 2*math.pi
    if (endAngle - startAngle) > 3*math.pi:
        endAngle -= 2*math.pi
    elif (endAngle - startAngle) < math.pi:
        endAngle += 2*math.pi
    diffAngle = endAngle - startAngle
    return startAngle, endAngle, diffAngle

def projectPointCloud(lidarData, ang_res_x, ang_res_y, N_SCAN, Horizon_SCAN, rangeMat):
    # Input:
    # rangeMat: N_SCAN*Horizon_SCCAN matrix init with values of -1
    # TODO: for carla we can get true rowIdn by simulator
    cloudSize = lidarData.shape[0]
    point = np.zeros((3, 1))
    for i in range(cloudSize):
        if lidarData[i, 3] == 0:
            continue
        point[0, 0] = lidarData[i, 0]
        point[1, 0] = lidarData[i, 1]
        point[2, 0] = lidarData[i, 2]

        horizonAngle = math.atan2(point[0, 0], point[1, 0]) * 180 / math.pi
        columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN
        if columnIdn >= Horizon_SCAN:
            columnIdn -= Horizon_SCAN
        if columnIdn < 0 or columnIdn >= Horizon_SCAN:
            continue

        # verticalAngle = math.atan2(point[2, 0], math.sqrt(point[0, 0]*point[0, 0] + point[1, 0]*point[1, 0])) * 180 / math.pi
        # rowIdn = verticalAngle / ang_res_y
        rowIdn = i // Horizon_SCAN
        if rowIdn < 0 or rowIdn >= N_SCAN:
            continue

        d = math.sqrt(point[0, 0]*point[0, 0] + point[1, 0]*point[1, 0] + point[2, 0]*point[2, 0])
        rangeMat[rowIdn, columnIdn] = d

def groundRemoval(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, groundMat, labelMat, rangeMat):
    for j in range(Horizon_SCAN):
        for i in range(groundScanInd):
            lowerInd = j + i * Horizon_SCAN
            upperInd = j + (i+1) * Horizon_SCAN
            if lidarData[lowerInd, 3] == 0 or lidarData[upperInd, 3] == 0:
                groundMat[i, j] = -1
                continue

            diffX = lidarData[upperInd, 0] - lidarData[lowerInd, 0]
            diffY = lidarData[upperInd, 1] - lidarData[lowerInd, 1]
            diffZ = lidarData[upperInd, 2] - lidarData[lowerInd, 2]

            angle = math.atan2(diffZ, math.sqrt(diffX*diffX + diffY*diffY)) * 180 / math.pi
            if abs(angle) <= 10:
                groundMat[i, j] = 1
                groundMat[i+1, j] = 1
    # labelMat == -2: useless point
    for i in range(N_SCAN):
        for j in range(Horizon_SCAN):
            if groundMat[i, j] == 1 or rangeMat[i, j] == -1:
                labelMat[i, j] == -2
    
def cloudSegmentation(N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat):
    for i in range(N_SCAN):
        for j in range(Horizon_SCAN):
            if labelMat[i, j] == 0:
                labelComponents(i, j, N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat)
    # TODO: handle the labelMat
            

def labelComponents(row, col, N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat, labelCount=labelCount):
    lineCountFlag = np.zeros(N_SCAN) == 1
    segmentAlphaX = ang_res_x / 180 * math.pi
    segmentAlphaY = ang_res_y / 180 * math.pi
    segmentTheta = 60 / 180 * math.pi

    queue = Queue()
    allqueue = Queue()
    queue.put((row, col))
    allqueue.put((row, col))
    # print(row,col)
    while queue.qsize() > 0:
        fromIndX, fromIndY = queue.get()
        labelMat[fromIndX, fromIndY] = labelCount
        for iter in neighbor(fromIndX, fromIndY):
            thisIndX = iter[0]
            thisIndY = iter[1]
            if thisIndX < 0 or thisIndX >= N_SCAN:
                continue
            if thisIndY < 0:
                thisIndY = Horizon_SCAN - 1
            if thisIndY >= Horizon_SCAN:
                thisIndY = 0
            
            if labelMat[thisIndX, thisIndY] != 0:
                continue
            d1 = max(rangeMat[fromIndX, fromIndY], rangeMat[thisIndX, thisIndY])
            d2 = min(rangeMat[fromIndX, fromIndY], rangeMat[thisIndX, thisIndY])

            # See "Fast Range Image-based Segmentation of Sparse 3D Laser Scans for Online Operation"
            if iter[0] == 0:
                alpha = segmentAlphaX
            else:
                alpha = segmentAlphaY
            angle = math.atan2(d2*math.sin(alpha), (d1 - d2*math.cos(alpha)))
            if (angle > segmentTheta):
                queue.put((thisIndX, thisIndY))
                labelMat[thisIndX, thisIndY] = labelCount
                lineCountFlag[thisIndX] = True
                allqueue.put((thisIndX, thisIndY))

    feasibleSegment = False
    if allqueue.qsize() >= 30:
        feasibleSegment = True
    elif allqueue.qsize() >= 5:
        lineCount = 0
        for i in range(N_SCAN):
            if lineCountFlag[i]:
                lineCount += 1
        if lineCount >= 3:
            feasibleSegment = True
    
    if feasibleSegment == True:
        labelCount += 1
    else:
        for i in range(allqueue.qsize()):
            labelMat[allqueue.queue[i][0], allqueue.queue[i][1]] = -1
    return labelCount

def neighbor(row, col):
    return (row-1, col), (row+1, col), (row, col-1), (row, col+1)

if __name__ == '__main__':
    lidarData = loadLidarData("G:/Carla/WindowsNoEditor/PythonAPI/output/lidar/2023-01-13-21-54-07-520741.txt")
    rangeMat = np.ones((N_SCAN, Horizon_SCAN)) * (-1)
    groundMat = np.zeros((N_SCAN, Horizon_SCAN))
    labelMat = np.zeros((N_SCAN, Horizon_SCAN))
    projectPointCloud(lidarData, ang_res_x, ang_res_y, N_SCAN, Horizon_SCAN, rangeMat)
    groundRemoval(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, groundMat, labelMat, rangeMat)
    cloudSegmentation(N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat)
    # print(rad2deg(startAngle), rad2deg(endAngle), rad2deg(diffAngle))
