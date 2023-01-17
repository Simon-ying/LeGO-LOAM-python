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

def extractFeatures(N_SCAN, startRingIndex, endRingIndex):
    conerPointsSharp = []
    cornerPointsLessSharp = []
    surfPointsFlat = []
    surfPointsLessFlat = []
    for i in range(N_SCAN):
        surfPointsLessFlat.clear()
        for j in range(6):
            sp = (startRingIndex[i] * (6-j) + endRingIndex[i] * j) / 6
            ep = (startRingIndex[i] * (5-j) + endRingIndex[i] * (j+1)) / 6 - 1

            if sp >= ep:
                continue
            
