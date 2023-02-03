from imageProjection import *
from featureAssociation import *
import os
if __name__ == "__main__":
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

    '''
    lidarCount = loadLidarData("lidarTestRaw_count.txt")
    lidarData_ori = loadLidarData("lidarTestRaw.txt")

    rangeMat = np.ones((N_SCAN, Horizon_SCAN)) * (-1)
    groundMat = np.zeros((N_SCAN, Horizon_SCAN))
    labelMat = np.zeros((N_SCAN, Horizon_SCAN))
    lidarData = projectPointCloud(lidarData_ori, ang_res_x, ang_res_y, N_SCAN, Horizon_SCAN, rangeMat)
    groundRemoval(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, groundMat, labelMat, rangeMat)
    startRingIndex, endRingIndex, outlierCloud, segmentedCloudGroundFlag, segmentedCloudColInd, segmentedCloudRange, segmentedCloud = cloudSegmentation(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, ang_res_x, ang_res_y, labelMat, rangeMat, groundMat)

    # dataList_seg = []
    # colorList_seg = []
    # for i in range(len(segmentedCloud)):
    #     if segmentedCloudGroundFlag[i]:
    #         colorList_seg.append(np.array([1,0,0]))
    #     else:
    #         colorList_seg.append(np.array([1,1,1]))
    #     dataList_seg.append(segmentedCloud[i])
    # dataList_seg = np.array(dataList_seg)
    # colorList_seg = np.array(colorList_seg)
    # dataList = removeEmptyData(lidarData)

    vis = False
    if vis:
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

    segmentedCloud = np.array(segmentedCloud)
    cloudCurvature = np.zeros(segmentedCloud.shape[0])
    cloudNeighborPicked = np.zeros(segmentedCloud.shape[0])
    cloudLabel = np.zeros(segmentedCloud.shape[0])
    cloudSmoothness = np.zeros((segmentedCloud.shape[0],2))

    # calculateSmoothness(segmentedCloud, segmentedCloudRange, cloudCurvature, cloudNeighborPicked, cloudLabel, cloudSmoothness)
    # markOccludePoints(segmentedCloud, segmentedCloudRange, segmentedCloudColInd, cloudNeighborPicked)
    '''

    frame = 1
    outputPath = "G:\Carla//CARLA_0.9.13//WindowsNoEditor//PythonAPI//_out//"
    img_files_path = outputPath + "camera//"
    gt_path = outputPath + "gt//"
    lidar_files_path = outputPath + "lidar//"

    lidar_files = os.listdir(lidar_files_path)
    laserCloudCornerLast = []
    laserCloudSurfLast = []
    transformCur = np.zeros(6)
    transformSum = np.zeros(6)
    for data in lidar_files:
        # print(data)
        lidarData_ori = loadLidarData(lidar_files_path+"/"+data)

        rangeMat = np.ones((N_SCAN, Horizon_SCAN)) * (-1)
        groundMat = np.zeros((N_SCAN, Horizon_SCAN))
        labelMat = np.zeros((N_SCAN, Horizon_SCAN))
        lidarData = projectPointCloud(lidarData_ori, ang_res_x, ang_res_y, N_SCAN, Horizon_SCAN, rangeMat)
        groundRemoval(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, groundMat, labelMat, rangeMat)
        startRingIndex, endRingIndex, outlierCloud, segmentedCloudGroundFlag, segmentedCloudColInd, segmentedCloudRange, segmentedCloud = cloudSegmentation(lidarData, N_SCAN, Horizon_SCAN, groundScanInd, ang_res_x, ang_res_y, labelMat, rangeMat, groundMat)

        segmentedCloud = np.array(segmentedCloud)

        feature = featureAccociation(segmentedCloud, segmentedCloudRange, segmentedCloudColInd, N_SCAN, startRingIndex, endRingIndex, segmentedCloudGroundFlag)
        # cloudCurvature = np.zeros(segmentedCloud.shape[0])
        # cloudNeighborPicked = np.zeros(segmentedCloud.shape[0])
        # cloudLabel = np.zeros(segmentedCloud.shape[0])
        # cloudSmoothness = np.zeros((segmentedCloud.shape[0],2))
        feature.calculateSmoothness()
        feature.markOccludePoints()
        cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat = feature.extractFeatures()

        cornerPointsSharpColor = np.array([[1,1,1]]*len(cornerPointsSharp))
        cornerPointsLessSharpColor = np.array([[1,1,1]]*len(cornerPointsLessSharp))
        surfPointsFlatColor = np.array([[0,1,0]]*len(surfPointsFlat))
        surfPointsLessFlatColor = np.array([[0,0,1]]*len(surfPointsLessFlat))
        cloud_extracted = np.r_[cornerPointsLessSharp, surfPointsLessFlat]
        color_extracted = np.r_[cornerPointsLessSharpColor, surfPointsLessFlatColor]
        # visulizeLiadarData(cloud_extracted, color_extracted)

        if frame == 1:
            laserCloudCornerLast = cornerPointsLessSharp
            laserCloudSurfLast = surfPointsLessFlat
            print("less Sharp number:", len(cornerPointsLessSharp), " less Surf number:", len(surfPointsLessFlat))
            print("Sharp number:", len(cornerPointsSharp), " Surf number:", len(surfPointsFlat))
            print("Current transform:", transformCur)
            print("Current position:", transformSum)
            frame += 1
            continue
        else:
            feature.updateTransformation(laserCloudCornerLast, laserCloudSurfLast, surfPointsFlat, cornerPointsSharp, transformCur)
            feature.interateTransformation(transformSum, transformCur)
            print("less Sharp number:", len(cornerPointsLessSharp), " less Surf number:", len(surfPointsLessFlat))
            print("Sharp number:", len(cornerPointsSharp), " Surf number:", len(surfPointsFlat))
            print("Current transform:", transformCur)
            print("Current position:", transformSum)
            frame += 1
            laserCloudCornerLast = cornerPointsLessSharp
            laserCloudSurfLast = surfPointsLessFlat
