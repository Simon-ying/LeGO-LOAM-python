'''
FeatureAssociation():
        nh("~")
        {
        // 订阅和发布各类话题
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &FeatureAssociation::laserCloudHandler, this);
        subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &FeatureAssociation::laserCloudInfoHandler, this);
        subOutlierCloud = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlierCloudHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imuHandler, this);

        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

        pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
        pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
        pubOutlierCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);
        
        initializationValue();
    }
'''

from dataHandler import loadLidarData
from dataHandler import visulizeLiadarData
from dataHandler import removeEmptyData

import numpy as np
import math
from math import sin, cos, atan2
from queue import Queue

def calculateSmoothness(segmentedCloud, segmentedCloudRange, cloudCurvature, cloudNeighborPicked, cloudLabel, cloudSmoothness):
    cloudSize = segmentedCloud.shape[0]
    for i in range(5, cloudSize - 5):
        diffRange = segmentedCloudRange[i-5] + segmentedCloudRange[i-4] \
                            + segmentedCloudRange[i-3] + segmentedCloudRange[i-2] \
                            + segmentedCloudRange[i-1] - segmentedCloudRange[i] * 10 \
                            + segmentedCloudRange[i+1] + segmentedCloudRange[i+2] \
                            + segmentedCloudRange[i+3] + segmentedCloudRange[i+4] \
                            + segmentedCloudRange[i+5]
        cloudCurvature[i] = diffRange*diffRange
        cloudNeighborPicked[i] = 0
        cloudLabel[i] = 0
        cloudSmoothness[i, 0] = cloudCurvature[i]
        cloudSmoothness[i, 1] = i

def markOccludePoints(segmentedCloud, segmentedCloudRange, segmentedCloudColInd, cloudNeighborPicked):
    cloudSize = segmentedCloud.shape[0]
    for i in range(5, cloudSize-6):
        depth1 = segmentedCloudRange[i]
        depth2 = segmentedCloudRange[i+1]
        columnDiff = abs(segmentedCloudColInd[i+1] - segmentedCloudColInd[i])

        if columnDiff < 10:
            if depth1 - depth2 > 0.3:
                cloudNeighborPicked[i-5] = 1
                cloudNeighborPicked[i-4] = 1
                cloudNeighborPicked[i-3] = 1
                cloudNeighborPicked[i-2] = 1
                cloudNeighborPicked[i-1] = 1
                cloudNeighborPicked[i] = 1
            elif depth2 - depth1 > 0.3:
                cloudNeighborPicked[i+6] = 1
                cloudNeighborPicked[i+5] = 1
                cloudNeighborPicked[i+4] = 1
                cloudNeighborPicked[i+3] = 1
                cloudNeighborPicked[i+2] = 1
                cloudNeighborPicked[i+1] = 1
        
        diff1 = abs(segmentedCloudRange[i-1] - segmentedCloudRange[i])
        diff2 = abs(segmentedCloudRange[i+1] - segmentedCloudRange[i])

        if diff1 > 0.02 * segmentedCloudRange[i-1] and diff2 > 0.02 * segmentedCloudRange[i]:
            cloudNeighborPicked[i] = 1

def extractFeatures(N_SCAN, startRingIndex, endRingIndex, segmentedCloud, segmentedCloudColInd, cloudSmoothness, cloudNeighborPicked, cloudCurvature, segmentedCloudGroundFlag, cloudLabel):
    cornerPointsSharp = []
    cornerPointsLessSharp = []
    surfPointsFlat = []
    surfPointsLessFlat = []
    surfPointsLessFlatScan = []
    edgeThreshold = 0 # to modify
    surfThreshold = 0
    # cloudSmoothness_sorted = np.argsort(cloudSmoothness, axis=0)
    for i in range(N_SCAN):
        surfPointsLessFlatScan.clear()
        for j in range(6):
            sp = (startRingIndex[i] * (6-j) + endRingIndex[i] * j) / 6
            ep = (startRingIndex[i] * (5-j) + endRingIndex[i] * (j+1)) / 6 - 1

            if sp >= ep:
                continue
            # sorted(cloudSmoothness) sort cloudSmoothness
            largestPickedNum = 0
            for k in range(ep, sp-1, -1):
                ind = cloudSmoothness[k, 1]
                if cloudNeighborPicked[ind] == 0 and cloudCurvature[ind] > edgeThreshold and not segmentedCloudGroundFlag[ind]:
                    largestPickedNum += 1
                    if largestPickedNum <= 2:
                        cloudLabel[ind] = 2
                        cornerPointsSharp.append(segmentedCloud[ind, :])
                        cornerPointsLessSharp.append(segmentedCloud[ind, :])
                    elif largestPickedNum <= 20:
                        cloudLabel[ind] = 1
                        cornerPointsSharp.append(segmentedCloud[ind, :])
                    else:
                        break

                    cloudNeighborPicked[ind] = 1
                    for l in range(1, 6):
                        columnDiff = abs(int(segmentedCloudColInd[ind+l] - segmentedCloudColInd[ind+l-1]))
                        if columnDiff > 10:
                            break
                        cloudNeighborPicked[ind+l] = 1
                    for l in range(-1, -6, -1):
                        columnDiff = abs(int(segmentedCloudColInd[ind+l] - segmentedCloudColInd[ind+l+1]))
                        if columnDiff > 10:
                            break
                        cloudNeighborPicked[ind+l] = 1
            
            smallestPickedNum = 0
            for k in range(sp, ep+1):
                ind = cloudSmoothness[k, 1]
                if cloudNeighborPicked[ind] == 0 and cloudCurvature[ind] < surfThreshold and segmentedCloudGroundFlag[ind]:
                    surfPointsFlat.append(segmentedCloud[ind, :])
                    smallestPickedNum += 1
                    if smallestPickedNum >= 4:
                        break
                    cloudNeighborPicked[ind] = 1

                    for l in range(1, 6):
                        columnDiff = abs(int(segmentedCloudColInd[ind+l] - segmentedCloudColInd[ind+l-1]))
                        if columnDiff > 10:
                            break
                        cloudNeighborPicked[ind+l] = 1
                    
                    for l in range(-1, -6, -1):
                        columnDiff = abs(int(segmentedCloudColInd[ind+l] - segmentedCloudColInd[ind+l-+]))
                        if columnDiff > 10:
                            break
                        cloudNeighborPicked[ind+l] = 1
            for k in range(qp, ep+1):
                if cloudLabel <= 0:
                    surfPointsLessFlatScan.append(segmentedCloud[k, :])
        # TODO: down size filter

def TransformToStart():
    return
def TransformToEnd():
    return
def AccumulateRotation(cx, cy, cz, lx, ly, lz, ox, oy, oz):
    srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx)
    ox = -asin(srx)

    srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
                     + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy)
    crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
                    - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx))
    oy = atan2(srycrx / cos(ox), crycrx / cos(ox))

    srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
                    + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz)
    crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
                    - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx))
    oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox))
    return ox, oy, oz

def rad2deg(radians):
    return radians * 180 / math.pi
def deg2rad(degrees):
    return degrees * math.pi / 180

def findCorrespondingCornerFeatures(iterCount, cornerPointsSharp):
    cornerPointsSharpNum = len(cornerPointsSharp)
    for i in range(cornerPointsSharpNum):
        TransformToStart()
        
        if iterCount % 5 == 0:
            continue