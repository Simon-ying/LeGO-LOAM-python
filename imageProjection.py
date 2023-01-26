# There are 4 main jobs in this file
# findStartEndAngle(): find the start and end angle for the lidar
# projectPointCloud(): find ground points and other points that are too far from lidar
# groundRemoval(): remove useless ground points
# cloudSegmentation(): Use method in Fast Range Image-based Segmentation of Sparse 3D Laser Scans for Online Operation
# problems: segmentation part can't more ensemble more than 30 points ? 
# Default parameter:
# upper-fov = 15 degrees
# lower-fov = -25 degrees
# N_SCAN = channels = 64
# range = 100 m
# points-per-second = 500000
# rotation_frequency = 20 Hz
# groundScanInd = 15 for Ouster OS1-64

# ang_res_x = 360*rotation_frequency*channels / points_per-second = 0.2304
# ang_res_y = (upper-fov - lower-fov) / (channels -1) = 40 / 63 = 0.635
# Horizon_SCAN = points_per-second / (rotation_frequency*channels) = 1562

# labelMat:
# -3 : empty points
# -2 : ground points
# -1 : not selected points
# >0 : selected points

from dataHandler import loadLidarData
from dataHandler import visulizeLiadarData
from dataHandler import removeEmptyData

import numpy as np
import math
from queue import Queue
import argparse

labelCount = 1
upper_fov = 15.0
lower_fov = -25
N_SCAN = 64
points_per_second = 500000
rotation_frequency = 20
groundScanInd = 37
uselessScanInd = 7

Horizon_SCAN = round(points_per_second / (rotation_frequency * N_SCAN))
ang_res_x = 360.0 / Horizon_SCAN
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

def findRowIdn(index, lidarData_count):
    sum = 0
    row = 0
    while sum <= index:
        sum += lidarData_count[row]
        row += 1
    return row-1

def projectPointCloud(lidarData, ang_res_x, ang_res_y, N_SCAN, Horizon_SCAN, rangeMat):
    # Input:
    # rangeMat: N_SCAN*Horizon_SCCAN matrix init with values of -1
    # TODO: for carla we can get true rowIdn by simulator
    cloudSize = lidarData.shape[0]
    dataList = np.zeros((N_SCAN*Horizon_SCAN, 4))
    # print(cloudSize, np.sum(lidarData_count), dataList.shape[0])

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

        verticalAngle = math.atan2(point[2, 0], math.sqrt(point[0, 0]*point[0, 0] + point[1, 0]*point[1, 0])) * 180 / math.pi
        rowIdn = round((15 - verticalAngle) / ang_res_y)
        # verify true lidar scan with the calculated lidar scan: OK
        # if findRowIdn(i, lidarData_count) != rowIdn:
        #     print("True row index = ", findRowIdn(i, lidarData_count), " Row index = ", rowIdn)
        if rowIdn < 0 or rowIdn >= N_SCAN:
            continue

        d = math.sqrt(point[0, 0]*point[0, 0] + point[1, 0]*point[1, 0] + point[2, 0]*point[2, 0])
        rangeMat[rowIdn, columnIdn] = d
        dataList[rowIdn * Horizon_SCAN + columnIdn, :] = lidarData[i, :]
    
    return dataList

def groundRemoval(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, groundMat, labelMat, rangeMat):
    for j in range(Horizon_SCAN):
        for i in range(N_SCAN-groundScanInd, N_SCAN):
            lowerRow = i - 1
            lowerInd = j + lowerRow * Horizon_SCAN
            upperInd = j + (i) * Horizon_SCAN
            if lidarData[upperInd, 3] == 0:
                groundMat[i, j] = -1
                continue
            for n in range(4):
                if lidarData[lowerInd, 3] != 0:
                    break
                else:
                    lowerRow -= 1
                    lowerInd = j + lowerRow * Horizon_SCAN

            diffX = lidarData[upperInd, 0] - lidarData[lowerInd, 0]
            diffY = lidarData[upperInd, 1] - lidarData[lowerInd, 1]
            diffZ = lidarData[upperInd, 2] - lidarData[lowerInd, 2]

            angle = math.atan2(diffZ, math.sqrt(diffX*diffX + diffY*diffY)) * 180 / math.pi
            if abs(angle) <= 10:
                groundMat[lowerRow, j] = 1
                groundMat[i, j] = 1
    # labelMat == -2: useless point
    for i in range(N_SCAN):
        for j in range(Horizon_SCAN):
            if groundMat[i, j] == 1 or rangeMat[i, j] == -1:
                labelMat[i, j] = -2
    
def cloudSegmentation(lidarData, N_SCAN, Horizon_SCAN,groundScanInd, ang_res_x, ang_res_y, labelMat, rangeMat, groundMat):
    labelCount = 1
    for i in range(N_SCAN):
        for j in range(Horizon_SCAN):
            if labelMat[i, j] == 0:
                if rangeMat[i, j] == -1:
                    labelMat[i, j] = -3
                    continue
                labelCount = labelComponents(i, j, N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat, labelCount)
    
    sizeOfSegCloud = 0
    startRingIndex = np.zeros(N_SCAN)
    endRingIndex = np.zeros(N_SCAN)
    outlierCloud = []
    segmentedCloudGroundFlag = []
    segmentedCloudColInd = []
    segmentedCloudRange = []
    segmentedCloud = []

    for i in range(N_SCAN):
        startRingIndex[i] = sizeOfSegCloud - 1 + 5
        for j in range(Horizon_SCAN):
            # remove the useless points arround the car 7=useless Scan
            if i > N_SCAN - 7:
                continue
            # Lack of enough points in cloud
            if labelMat[i, j] == -1:
                if i < N_SCAN - groundScanInd and j % 5 == 0:
                    outlierCloud.append(lidarData[i*Horizon_SCAN + j, :])
                    continue
                else:
                    continue
            if labelMat[i, j] > 0 or groundMat[i, j] == 1:
                if groundMat[i, j] == 1:
                    if j % 5 != 0 and j > 5 and j < Horizon_SCAN - 5:
                        continue
                segmentedCloudGroundFlag.append(groundMat[i, j] == 1)
                segmentedCloudColInd.append(j)
                segmentedCloudRange.append(rangeMat[i, j])
                segmentedCloud.append(np.r_[lidarData[i*Horizon_SCAN + j, :-1],i])
                sizeOfSegCloud += 1
            endRingIndex[i] = sizeOfSegCloud - 1 - 5

            

    return startRingIndex, endRingIndex, outlierCloud, segmentedCloudGroundFlag, segmentedCloudColInd, segmentedCloudRange, segmentedCloud
            

def labelComponents(row, col, N_SCAN, Horizon_SCAN, ang_res_x, ang_res_y, labelMat, rangeMat, labelCount=1):
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
            if d2 == -1:
                continue

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
        # print("cloud count: ", allqueue.qsize())
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
    # TODO: use lidardataraw
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--flag',
        default=0,
        type=int,
        help='0: initial cloud, 1: initial cloud with colors, 2: segmented cloud, 3: ground points, 4: remove ground points')
    argparser.add_argument(
        '--data',
        default=0,
        type=int,
        help='0: initial data, 1: another data')
    args = argparser.parse_args()

    lidarCount = loadLidarData("lidarTestRaw_count.txt")
    lidarData_ori = loadLidarData("lidarTestRaw.txt")
    if args.data == 1:
        lidarCount = loadLidarData("lidarTestRaw_count3.txt")
        lidarData_ori = loadLidarData("lidarTestRaw3.txt")

    rangeMat = np.ones((N_SCAN, Horizon_SCAN)) * (-1)
    groundMat = np.zeros((N_SCAN, Horizon_SCAN))
    labelMat = np.zeros((N_SCAN, Horizon_SCAN))
    lidarData = projectPointCloud(lidarData_ori, ang_res_x, ang_res_y, N_SCAN, Horizon_SCAN, rangeMat, lidarCount)
    groundRemoval(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, groundMat, labelMat, rangeMat)
    startRingIndex, endRingIndex, outlierCloud, segmentedCloudGroundFlag, segmentedCloudColInd, segmentedCloudRange, segmentedCloud = cloudSegmentation(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, ang_res_x, ang_res_y, labelMat, rangeMat, groundMat)

    # groundMat 1: ground points

    

    dataList_seg = []
    colorList_seg = []
    for i in range(len(segmentedCloud)):
        if segmentedCloudGroundFlag[i]:
            colorList_seg.append(np.array([1,0,0]))
        else:
            colorList_seg.append(np.array([1,1,1]))
        dataList_seg.append(segmentedCloud[i])
    dataList_seg = np.array(dataList_seg)
    colorList_seg = np.array(colorList_seg)
    # dataList = removeEmptyData(lidarData)

    if args.flag == 0:
        print("All points: ", np.sum(lidarCount))
        visulizeLiadarData(removeEmptyData(lidarData_ori))
        
    elif args.flag == 2:
        visulizeLiadarData(dataList_seg, colorList_seg)
    elif args.flag == 3:
        for i in range(N_SCAN):
            for j in range(Horizon_SCAN):
                if labelMat[i, j] != -2:
                    lidarData[i*Horizon_SCAN+j, 3] = 0
        visulizeLiadarData(removeEmptyData(lidarData))
    elif args.flag == 4:
        for i in range(N_SCAN):
            for j in range(Horizon_SCAN):
                if labelMat[i, j] == -2:
                    lidarData[i*Horizon_SCAN+j, 3] = 0
        visulizeLiadarData(removeEmptyData(lidarData))
    else:
        dataList_other = []
        colorList_other = []
        for i in range(N_SCAN):
            for j in range(Horizon_SCAN):
                if labelMat[i, j] == -1:
                    if lidarData[i*Horizon_SCAN + j, 3] == 0:
                        continue
                    dataList_other.append(lidarData[i*Horizon_SCAN + j, :])
                    colorList_other.append(np.array([0,1,0]))
        dataList_other = np.array(dataList_other)
        colorList_other = np.array(colorList_other)
        dataList_other = np.r_[dataList_other, dataList_seg]
        colorList_other = np.r_[colorList_other, colorList_seg]
        # print(dataList_other.shape, colorList_other.shape)
        visulizeLiadarData(dataList_other, colorList_other)
    
    # print(startRingIndex)
    # print(endRingIndex)
