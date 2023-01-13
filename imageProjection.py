# There are 4 main jobs in this file
# findStartEndAngle(): find the start and end angle for the lidar
# projectPointCloud(): find ground points and other points that are too far from lidar
# groundRemoval(): remove useless ground points
# cloudSegmentation(): Use method in Fast Range Image-based Segmentation of Sparse 3D Laser Scans for Online Operation

from dataHandler import loadLidarData

import numpy as np
import math
from queue import Queue

labelCount = 1

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
        point[0, 0] = lidarData[i, 0]
        point[1, 0] = lidarData[i, 1]
        point[2, 0] = lidarData[i, 2]
        verticalAngle = math.atan2(point[2, 0], math.sqrt(point[0, 0]*point[0, 0] + point[1, 0]*point[1, 0])) * 180 / math.pi
        rowIdn = verticalAngle / ang_res_y
        if rowIdn < 0 or rowIdn >= N_SCAN:
            continue

        horizonAngle = math.atan2(point[0, 0], point[1, 0]) * 180 / math.pi
        columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN
        if columnIdn >= Horizon_SCAN:
            columnIdn -= Horizon_SCAN
        if columnIdn < 0 or columnIdn >= Horizon_SCAN:
            continue
        range = math.sqrt(point[0, 0]*point[0, 0] + point[1, 0]*point[1, 0], point[2, 0]*point[2, 0])
        rangeMat[rowIdn, columnIdn] = range

def groundRemoval(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, groundMat, labelMat, rangeMat):
    for j in range(Horizon_SCAN):
        for i in range(groundScanInd):
            lowerInd = j + i * Horizon_SCAN
            upperInd = j + (i+1) * Horizon_SCAN
            if lidarData[lowerInd, 3] == -1 or lidarData[upperInd, 3] == -1:
                groundMat[i, j] = -1
                continue

            diffX = lidarData[upperInd, 0] - lidarData[lowerInd, 0]
            diffY = lidarData[upperInd, 1] - lidarData[lowerInd, 1]
            diffZ = lidarData[upperInd, 2] - lidarData[lowerInd, 2]

            angle = math.atan2(diffZ, math.sqrt(diffX*diffX + diffY*diffY)) * 180 / math.pi
            if abs(angle) <= 10:
                groundMat[i, j] = 1
                groundMat[i+1, j] = 1
    
    for i in range(N_SCAN):
        for j in range(Horizon_SCAN):
            if groundMat[i, j] == 1 or rangeMat[i, j] == -1:
                labelMat[i, j] == -1
    
def cloudSegmentation(N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat):
    for i in range(N_SCAN):
        for j in range(Horizon_SCAN):
            if labelMat[i, j] == 0:
                labelComponents(i, j, N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat)
    # TODO: handle the labelMat
            

def labelComponents(row, col, N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat):
    lineCountFlag = np.zeros((1, N_SCAN)) == 1
    segmentAlphaX = ang_res_x / 180 * math.pi
    segmentAlphaY = ang_res_y / 180 * math.pi
    segmentTheta = 60 / 180 * math.pi

    queue = Queue()
    allqueue = Queue()
    queue.put((row, col))
    allqueue.put((row, col))
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
                thisIndX = 0
            
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


def neighbor(row, col):
    return (row-1, col), (row+1, col), (row, col-1), (row, col+1)

if __name__ == '__main__':
    lidarData = loadLidarData('test.txt')
    startAngle, endAngle, diffAngle = findStartEndAngle(lidarData)

    # print(rad2deg(startAngle), rad2deg(endAngle), rad2deg(diffAngle))
    labelMat = np.zeros((3,4))
    labelComponents(lidarData, 1, 3, labelMat)