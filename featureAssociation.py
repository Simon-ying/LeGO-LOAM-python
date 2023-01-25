from dataHandler import loadLidarData
from dataHandler import visulizeLiadarData
from dataHandler import removeEmptyData

import numpy as np
import math
from math import sin, cos, atan2, asin
from queue import Queue
import copy
import cv2 as cv

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
    edgeThreshold = 0.1
    surfThreshold = 0.1
    # cloudSmoothness_sorted = np.argsort(cloudSmoothness, axis=0)
    for i in range(N_SCAN):
        surfPointsLessFlatScan.clear()
        for j in range(6):
            sp = (startRingIndex[i] * (6-j) + endRingIndex[i] * j) / 6
            ep = (startRingIndex[i] * (5-j) + endRingIndex[i] * (j+1)) / 6 - 1

            if sp >= ep:
                continue
            sorted_args = np.argsort(cloudSmoothness[sp:ep+1, 0], axis=0)[::-1]
            sorted_args += sp

            largestPickedNum = 0
            for k in range(ep, sp-1, -1):
                ind = cloudSmoothness[sorted_args[k-sp], 1]
                if cloudNeighborPicked[ind] == 0 and cloudCurvature[ind] > edgeThreshold and not segmentedCloudGroundFlag[ind]:
                    largestPickedNum += 1
                    if largestPickedNum <= 2:
                        cloudLabel[ind] = 2
                        cornerPointsSharp.append(segmentedCloud[ind, :])
                        cornerPointsLessSharp.append(segmentedCloud[ind, :])
                    elif largestPickedNum <= 20:
                        cloudLabel[ind] = 1
                        cornerPointsLessSharp.append(segmentedCloud[ind, :])
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
                ind = cloudSmoothness[sorted_args[k-sp], 1]
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
                        columnDiff = abs(int(segmentedCloudColInd[ind+l] - segmentedCloudColInd[ind+l+1]))
                        if columnDiff > 10:
                            break
                        cloudNeighborPicked[ind+l] = 1
            for k in range(sp, ep+1):
                if cloudLabel <= 0:
                    surfPointsLessFlatScan.append(segmentedCloud[k, :])
        # TODO: downsize filter
        surfPointsLessFlat += copy.deepcopy(surfPointsLessFlatScan)
    return cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat
    
def AccumulateRotation(cx, cy, cz, lx, ly, lz):
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
def d(point1, point2):
    diffX = point1[0] - point2[0]
    diffY = point1[1] - point2[1]
    diffZ = point1[2] - point2[2]
    return diffX*diffX + diffY*diffY + diffZ*diffZ

# laserCloudCornerLast : [x, y, z, scan]
# pointSearchCornerInd1&2 : initialize with -1
def findCorrespondingCornerFeatures(iterCount, cornerPointsSharp, laserCloudCornerLast):
    cornerPointsSharpNum = len(cornerPointsSharp)
    nearestFeatureSearchSqDist = 25
    laserCloudOri = []
    coeffSel = []
    pointSearchCornerInd1 = np.ones(cornerPointsSharpNum) * (-1)
    pointSearchCornerInd2 = np.ones(cornerPointsSharpNum) * (-1)

    for i in range(cornerPointsSharpNum):
        # pointSel == cornerPointsShape[i, :]
        pointSel = cornerPointsSharp[i]
        if iterCount % 5 == 0:
            # TODO: KD=Tree for laser cloud corner&surf last            
            # Find nearest point
            ind = 0
            dis = math.inf
            for i in range(len(laserCloudCornerLast)):
                temp_dis = d(pointSel, laserCloudCornerLast[i])
                if temp_dis < dis:
                    dis = temp_dis
                    ind = i
            closestPointInd = -1
            minPointInd2 = -1

            if dis < nearestFeatureSearchSqDist:
                closestPointInd = ind
                closestPointScan = laserCloudCornerLast[closestPointInd][3]
                pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist

                for j in range(closestPointInd+1, cornerPointsSharpNum):
                    if laserCloudCornerLast[j][3] > closestPointScan + 2.5:
                        break
                    pointSqDis = (laserCloudCornerLast[j][0] - pointSel[0]) * (laserCloudCornerLast[j][0] - pointSel[0]) + \
                                 (laserCloudCornerLast[j][1] - pointSel[1]) * (laserCloudCornerLast[j][1] - pointSel[1]) + \
                                 (laserCloudCornerLast[j][2] - pointSel[2]) * (laserCloudCornerLast[j][2] - pointSel[2])
                    if laserCloudCornerLast[j][3] > closestPointScan:
                        if pointSqDis < minPointInd2:
                            minPointSqDis2 = pointSqDis
                            minPointInd2 = j
                
                for j in range(closestPointInd-1, -1,-1):
                    if laserCloudCornerLast[j][3] < closestPointScan - 2.5:
                        break
                    pointSqDis = (laserCloudCornerLast[j][0] - pointSel[0]) * (laserCloudCornerLast[j][0] - pointSel[0]) + \
                                 (laserCloudCornerLast[j][1] - pointSel[1]) * (laserCloudCornerLast[j][1] - pointSel[1]) + \
                                 (laserCloudCornerLast[j][2] - pointSel[2]) * (laserCloudCornerLast[j][2] - pointSel[2])
                    if laserCloudCornerLast[j][3] < closestPointScan:
                        if pointSqDis < minPointInd2:
                            minPointSqDis2 = pointSqDis
                            minPointInd2 = j
                pointSearchCornerInd1[i] = closestPointInd
                pointSearchCornerInd2[i] = minPointInd2

            if pointSearchCornerInd2[i] >= 0:
                tripod1 = laserCloudCornerLast[pointSearchCornerInd1[i]]
                tripod2 = laserCloudCornerLast[pointSearchCornerInd2[i]]
                coeff = np.zeros((4,1))
                # TODO: understand coeff
                x0 = pointSel[0]
                y0 = pointSel[1]
                z0 = pointSel[2]
                x1 = tripod1[0]
                y1 = tripod1[1]
                z1 = tripod1[2]
                x2 = tripod2[0]
                y2 = tripod2[1]
                z2 = tripod2[2]

                # (X-Y1)×(X-Y2) = m11*k + m22*j + m33*i i=(1,0,0), j=(0,1,0), k=(0,0,1)
                m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))

                # a012 = |(X-Y1)×(X-Y2)|
                a012 = math.sqrt(m11 * m11  + m22 * m22 + m33 * m33)

                # l12 = |Y1-Y2|
                l12 = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2))

                la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12

                lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12

                lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12

                ld2 = a012 / l12

                s = 1
                if iterCount >= 5:
                    s = 1 - 1.8 * math.fabs(ld2)
    

                if s > 0.1 and ld2 != 0:
                    coeff[0] = s * la 
                    coeff[1] = s * lb
                    coeff[2] = s * lc
                    coeff[3] = s * ld2
                  
                    laserCloudOri.append(cornerPointsSharp[i])
                    coeffSel.append(coeff)
    return laserCloudOri, coeffSel


def findCorrespondingSurfFeatures(iterCount, surfPointsFlat, laserCloudSurfLast):
    surfPointsSharpNum = len(surfPointsFlat)
    nearestFeatureSearchSqDist = 25
    laserCloudOri = []
    coeffSel = []
    pointSearchSurfInd1 = np.ones(surfPointsSharpNum) * (-1)
    pointSearchSurfInd2 = np.ones(surfPointsSharpNum) * (-1)
    pointSearchSurfInd3 = np.ones(surfPointsSharpNum) * (-1)

    for i in range(surfPointsSharpNum):
        # pointSel == cornerPointsShape[i, :]
        pointSel = surfPointsFlat[i]
        if iterCount % 5 == 0:
            # TODO: KD=Tree for laser cloud corner&surf last            
            # Find nearest point
            ind = 0
            dis = math.inf
            for i in range(len(laserCloudSurfLast)):
                temp_dis = d(pointSel, laserCloudSurfLast[i])
                if temp_dis < dis:
                    dis = temp_dis
                    ind = i
            closestPointInd = -1
            minPointInd2 = -1
            minPointInd3 = -1

            if dis < nearestFeatureSearchSqDist:
                closestPointInd = ind
                closestPointScan = laserCloudSurfLast[closestPointInd][3]
                pointSqDis, minPointSqDis2, minPointSqDis3 = nearestFeatureSearchSqDist

                for j in range(closestPointInd+1, surfPointsSharpNum):
                    if laserCloudSurfLast[j][3] > closestPointScan + 2.5:
                        break
                    pointSqDis = (laserCloudSurfLast[j][0] - pointSel[0]) * (laserCloudSurfLast[j][0] - pointSel[0]) + \
                                 (laserCloudSurfLast[j][1] - pointSel[1]) * (laserCloudSurfLast[j][1] - pointSel[1]) + \
                                 (laserCloudSurfLast[j][2] - pointSel[2]) * (laserCloudSurfLast[j][2] - pointSel[2])
                    if laserCloudSurfLast[j][3] <= closestPointScan:
                        if pointSqDis < minPointInd2:
                            minPointSqDis2 = pointSqDis
                            minPointInd2 = j
                    else:
                        if pointSqDis < minPointSqDis3:
                            minPointSqDis3 = pointSqDis
                            minPointInd3 = j
                
                for j in range(closestPointInd-1, -1, -1):
                    if laserCloudSurfLast[j][3] > closestPointScan - 2.5:
                        break
                    pointSqDis = (laserCloudSurfLast[j][0] - pointSel[0]) * (laserCloudSurfLast[j][0] - pointSel[0]) + \
                                 (laserCloudSurfLast[j][1] - pointSel[1]) * (laserCloudSurfLast[j][1] - pointSel[1]) + \
                                 (laserCloudSurfLast[j][2] - pointSel[2]) * (laserCloudSurfLast[j][2] - pointSel[2])
                    if laserCloudSurfLast[j][3] >= closestPointScan:
                        if pointSqDis < minPointInd2:
                            minPointSqDis2 = pointSqDis
                            minPointInd2 = j
                    else:
                        if pointSqDis < minPointSqDis3:
                            minPointSqDis3 = pointSqDis
                            minPointInd3 = j
                pointSearchSurfInd1[i] = closestPointInd
                pointSearchSurfInd2[i] = minPointInd2
                pointSearchSurfInd3[i] = minPointInd3
            if pointSearchSurfInd2[i] >= 0 and pointSearchSurfInd3[i] >= 0:
                tripod1 = laserCloudSurfLast[pointSearchSurfInd1[i]]
                tripod2 = laserCloudSurfLast[pointSearchSurfInd2[i]]
                tripod3 = laserCloudSurfLast[pointSearchSurfInd3[i]]
                coeff = np.zeros((4,1))

                pa = (tripod2[1] - tripod1[1]) * (tripod3[2] - tripod1[2]) - (tripod3[1] - tripod1[1]) * (tripod2[2] - tripod1[2])
                pb = (tripod2[2] - tripod1[2]) * (tripod3[0] - tripod1[0]) - (tripod3[2] - tripod1[2]) * (tripod2[0] - tripod1[0])
                pc = (tripod2[0] - tripod1[0]) * (tripod3[1] - tripod1[1]) - (tripod3[0] - tripod1[0]) * (tripod2[1] - tripod1[1])
                pd = -(pa * tripod1[0] + pb * tripod1[1] + pc * tripod1[2])

                ps = math.sqrt(pa * pa + pb * pb + pc * pc)

                pa /= ps
                pb /= ps
                pc /= ps
                pd /= ps

                pd2 = pa * pointSel[0] + pb * pointSel[1] + pc * pointSel[2] + pd

                s = 1
                if iterCount >= 5:
                    s = 1 - 1.8 * math.fabs(pd2) / math.sqrt(math.sqrt(pointSel[0] * pointSel[0]
                            + pointSel[1] * pointSel[1] + pointSel[2] * pointSel[2]))

                if s > 0.1 and pd2 != 0:
                    coeff[0] = s * pa
                    coeff[1] = s * pb
                    coeff[2] = s * pc
                    coeff[3] = s * pd2

                    laserCloudOri.append(surfPointsFlat[i])
                    coeffSel.append(coeff)
    return laserCloudOri, coeffSel


def calculateTransformationSurf(iterCount, laserCloudOri, coeffSel, transformCur):
    # transformCur = np.zeros((6,1), np.float32)
    pointSelNum = len(laserCloudOri)

    matA = np.zeros((pointSelNum, 3), np.float32)
    matAt = np.zeros((3, pointSelNum), np.float32)
    matAtA = np.zeros((3, 3), np.float32)
    matB = np.zeros((pointSelNum, 1), np.float32)
    matAtB = np.zeros((3, 1), np.float32)
    matX = np.zeros((3, 1), np.float32)

    srx = sin(transformCur[0])
    crx = cos(transformCur[0])
    sry = sin(transformCur[1])
    cry = cos(transformCur[1])
    srz = sin(transformCur[2])
    crz = cos(transformCur[2])
    tx = transformCur[3]
    ty = transformCur[4]
    tz = transformCur[5]

    a1 = crx*sry*srz
    a2 = crx*crz*sry
    a3 = srx*sry
    a4 = tx*a1 - ty*a2 - tz*a3
    a5 = srx*srz
    a6 = crz*srx
    a7 = ty*a6 - tz*crx - tx*a5
    a8 = crx*cry*srz
    a9 = crx*cry*crz
    a10 = cry*srx
    a11 = tz*a10 + ty*a9 - tx*a8

    b1 = -crz*sry - cry*srx*srz
    b2 = cry*crz*srx - sry*srz
    b5 = cry*crz - srx*sry*srz
    b6 = cry*srz + crz*srx*sry

    c1 = -b6
    c2 = b5
    c3 = tx*b6 - ty*b5
    c4 = -crx*crz
    c5 = crx*srz
    c6 = ty*c5 + tx*-c4
    c7 = b2
    c8 = -b1
    c9 = tx*-b2 - ty*-b1

    for i in range(pointSelNum):
        pointOri = laserCloudOri[i]
        coeff = coeffSel[i]

        arx = (-a1*pointOri[0] + a2*pointOri[1] + a3*pointOri[2] + a4) * coeff[0] 
        + (a5*pointOri[0] - a6*pointOri[1] + crx*pointOri[2] + a7) * coeff[1]
        + (a8*pointOri[0] - a9*pointOri[1] - a10*pointOri[2] + a11) * coeff[2]

        arz = (c1*pointOri[0] + c2*pointOri[1] + c3) * coeff[0]
        + (c4*pointOri[0] - c5*pointOri[1] + c6) * coeff[1]
        + (c7*pointOri[0] + c8*pointOri[1] + c9) * coeff[2]

        aty = -b6 * coeff[0] + c4 * coeff[1] + b2 * coeff[2]

        d2 = coeff[3]

        matA[i, 0] = arx
        matA[i, 1] = arz
        matA[i, 2] = aty
        matB[i, 0] = -0.05 * d2

    matAt = cv.transpose(matA)
    matAtA = matAt.dot(matA)
    matAtB = matAt.dot(matB)
    matX = cv.solve(matAtA, matAtB, cv.DECOMP_QR)

    if iterCount == 0:
        # matE = np.zeros((1, 3), np.float32)
        # matV = np.zeros((3, 3), np.float32)
        # matV2 = np.zeros((3, 3), np.float32)

        eigens = cv.eigen(matAt)
        matE = eigens[0]
        matV = eigens[1]
        matV2 = copy.copy(matV)

        isDegenerate = False
        eignThre = [10, 10, 10]
        for i in range(2, -1, -1):
            if matE[0, i] < eignThre[i]:
                for j in range(3):
                    matV2[i, j] = 0
                isDegenerate = True
            else:
                break
        matP = np.linalg.inv(matV).dot(matV2)
    
    if isDegenerate:
        matX2 = copy.copy(matX)
        matX = matP.dot(matX2)

    transformCur[0] += matX[0, 0]
    transformCur[2] += matX[1, 0]
    transformCur[4] += matX[2, 0]

    for i in range(6):
        if np.isnan(transformCur[i]):
            transformCur[i]=0

    deltaR = math.sqrt(rad2deg(matX[0, 0])**2 + rad2deg(matX[1, 0])**2)
    deltaT = math.sqrt((matX[2, 0]*100)**2)

    if deltaR < 0.1 and deltaT < 0.1:
        return False
    return True


def calculateTransformationCorner(iterCount, laserCloudOri, coeffSel, transformCur):
    # transformCur = np.zeros((6,1), np.float32)
    pointSelNum = len(laserCloudOri)

    matA = np.zeros((pointSelNum, 3), np.float32)
    matAt = np.zeros((3, pointSelNum), np.float32)
    matAtA = np.zeros((3, 3), np.float32)
    matB = np.zeros((pointSelNum, 1), np.float32)
    matAtB = np.zeros((3, 1), np.float32)
    matX = np.zeros((3, 1), np.float32)

    srx = sin(transformCur[0])
    crx = cos(transformCur[0])
    sry = sin(transformCur[1])
    cry = cos(transformCur[1])
    srz = sin(transformCur[2])
    crz = cos(transformCur[2])
    tx = transformCur[3]
    ty = transformCur[4]
    tz = transformCur[5]

    b1 = -crz*sry - cry*srx*srz
    b2 = cry*crz*srx - sry*srz
    b3 = crx*cry
    b4 = tx*-b1 + ty*-b2 + tz*b3
    b5 = cry*crz - srx*sry*srz
    b6 = cry*srz + crz*srx*sry
    b7 = crx*sry
    b8 = tz*b7 - ty*b6 - tx*b5

    c5 = crx*srz

    for i in range(pointSelNum):
        pointOri = laserCloudOri[i]
        coeff = coeffSel[i]

        ary = (b1*pointOri[0] + b2*pointOri[1] - b3*pointOri.z + b4) * coeff[0]
        + (b5*pointOri[0] + b6*pointOri[1] - b7*pointOri.z + b8) * coeff.z

        atx = -b5 * coeff[0] + c5 * coeff[1] + b1 * coeff.z

        atz = b7 * coeff[0] - srx * coeff[1] - b3 * coeff.z

        d2 = coeff[3]

        matA[i, 0] = ary
        matA[i, 1] = atx
        matA[i, 2] = atz
        matB[i, 0] = -0.05 * d2

    matAt = cv.transpose(matA)
    matAtA = matAt.dot(matA)
    matAtB = matAt.dot(matB)
    matX = cv.solve(matAtA, matAtB, cv.DECOMP_QR)

    if iterCount == 0:
        # matE = np.zeros((1, 3), np.float32)
        # matV = np.zeros((3, 3), np.float32)
        # matV2 = np.zeros((3, 3), np.float32)

        eigens = cv.eigen(matAt)
        matE = eigens[0]
        matV = eigens[1]
        matV2 = copy.copy(matV)

        isDegenerate = False
        eignThre = [10, 10, 10]
        for i in range(2, -1, -1):
            if matE[0, i] < eignThre[i]:
                for j in range(3):
                    matV2[i, j] = 0
                isDegenerate = True
            else:
                break
        matP = np.linalg.inv(matV).dot(matV2)
    
    if isDegenerate:
        matX2 = copy.copy(matX)
        matX = matP.dot(matX2)

    transformCur[1] += matX[0, 0]
    transformCur[3] += matX[1, 0]
    transformCur[5] += matX[2, 0]

    for i in range(6):
        if np.isnan(transformCur[i]):
            transformCur[i]=0

    deltaR = math.sqrt(rad2deg(matX[0, 0])**2)
    deltaT = math.sqrt((matX[1, 0]*100)**2 + (matX[2, 0]*100)**2)
    
    if deltaR < 0.1 and deltaT < 0.1:
        return False
    return True


def calculateTransformation(iterCount, laserCloudOri, coeffSel, transformCur):
    # transformCur = np.zeros((6,1), np.float32)
    pointSelNum = len(laserCloudOri)

    matA = np.zeros((pointSelNum, 6), np.float32)
    matAt = np.zeros((6, pointSelNum), np.float32)
    matAtA = np.zeros((6, 6), np.float32)
    matB = np.zeros((pointSelNum, 1), np.float32)
    matAtB = np.zeros((6, 1), np.float32)
    matX = np.zeros((6, 1), np.float32)

    srx = sin(transformCur[0])
    crx = cos(transformCur[0])
    sry = sin(transformCur[1])
    cry = cos(transformCur[1])
    srz = sin(transformCur[2])
    crz = cos(transformCur[2])
    tx = transformCur[3]
    ty = transformCur[4]
    tz = transformCur[5]

    a1 = crx*sry*srz
    a2 = crx*crz*sry
    a3 = srx*sry
    a4 = tx*a1 - ty*a2 - tz*a3
    a5 = srx*srz
    a6 = crz*srx
    a7 = ty*a6 - tz*crx - tx*a5
    a8 = crx*cry*srz
    a9 = crx*cry*crz
    a10 = cry*srx
    a11 = tz*a10 + ty*a9 - tx*a8

    b1 = -crz*sry - cry*srx*srz
    b2 = cry*crz*srx - sry*srz
    b3 = crx*cry
    b4 = tx*-b1 + ty*-b2 + tz*b3
    b5 = cry*crz - srx*sry*srz
    b6 = cry*srz + crz*srx*sry
    b7 = crx*sry
    b8 = tz*b7 - ty*b6 - tx*b5

    c1 = -b6
    c2 = b5
    c3 = tx*b6 - ty*b5
    c4 = -crx*crz
    c5 = crx*srz
    c6 = ty*c5 + tx*-c4
    c7 = b2
    c8 = -b1
    c9 = tx*-b2 - ty*-b1

    for i in range(pointSelNum):
        pointOri = laserCloudOri[i]
        coeff = coeffSel[i]

        arx = (-a1*pointOri[0] + a2*pointOri[1] + a3*pointOri[2] + a4) * coeff[0] 
        + (a5*pointOri[0] - a6*pointOri[1] + crx*pointOri[2] + a7) * coeff[1]
        + (a8*pointOri[0] - a9*pointOri[1] - a10*pointOri[2] + a11) * coeff[2]

        ary = (b1*pointOri[0] + b2*pointOri[1] - b3*pointOri.z + b4) * coeff[0]
        + (b5*pointOri[0] + b6*pointOri[1] - b7*pointOri.z + b8) * coeff.z

        arz = (c1*pointOri[0] + c2*pointOri[1] + c3) * coeff[0]
        + (c4*pointOri[0] - c5*pointOri[1] + c6) * coeff[1]
        + (c7*pointOri[0] + c8*pointOri[1] + c9) * coeff[2]

        atx = -b5 * coeff[0] + c5 * coeff[1] + b1 * coeff.z

        aty = -b6 * coeff[0] + c4 * coeff[1] + b2 * coeff[2]

        atz = b7 * coeff[0] - srx * coeff[1] - b3 * coeff.z

        d2 = coeff[3]

        matA[i, 0] = arx
        matA[i, 1] = ary
        matA[i, 2] = arz
        matA[i, 3] = atx
        matA[i, 4] = aty
        matA[i, 5] = atz
        matB[i, 0] = -0.05 * d2

    matAt = cv.transpose(matA)
    matAtA = matAt.dot(matA)
    matAtB = matAt.dot(matB)
    matX = cv.solve(matAtA, matAtB, cv.DECOMP_QR)

    if iterCount == 0:
        # matE = np.zeros((1, 6), np.float32)
        # matV = np.zeros((6, 6), np.float32)
        # matV2 = np.zeros((6, 6), np.float32)

        eigens = cv.eigen(matAt)
        matE = eigens[0]
        matV = eigens[1]
        matV2 = copy.copy(matV)

        isDegenerate = False
        eignThre = [10, 10, 10, 10, 10, 10]
        for i in range(5, -1, -1):
            if matE[0, i] < eignThre[i]:
                for j in range(6):
                    matV2[i, j] = 0
                isDegenerate = True
            else:
                break
        matP = np.linalg.inv(matV).dot(matV2)
    
    if isDegenerate:
        matX2 = copy.copy(matX)
        matX = matP.dot(matX2)

    transformCur[0] += matX[0, 0]
    transformCur[1] += matX[1, 0]
    transformCur[2] += matX[2, 0]
    transformCur[3] += matX[3, 0]
    transformCur[4] += matX[4, 0]
    transformCur[5] += matX[5, 0]

    for i in range(6):
        if np.isnan(transformCur[i]):
            transformCur[i]=0

    deltaR = math.sqrt(rad2deg(matX[0, 0])**2 + rad2deg(matX[1, 0])**2 + rad2deg(matX[2, 0])**2)
    deltaT = math.sqrt((matX[3, 0]*100)**2 + (matX[4, 0]*100)**2 + (matX[5, 0]*100)**2)
    
    if deltaR < 0.1 and deltaT < 0.1:
        return False
    return True

def updateTransformation(laserCloudCornerLast, laserCloudSurfLast, surfPointsFlat, cornerPointSharp, transformCur):
    if len(laserCloudCornerLast) < 10 or len(laserCloudSurfLast) < 100:
        return
    for iterCount1 in range(25):
        laserCloudOri, coeffSel = findCorrespondingSurfFeatures(iterCount1, surfPointsFlat, laserCloudSurfLast)
        if len(laserCloudOri) < 10:
            continue
        if calculateTransformationSurf(iterCount1, laserCloudOri, coeffSel, transformCur) == False:
            break
    
    for iterCount2 in range(25):
        laserCloudOri, coeffSel = findCorrespondingCornerFeatures(iterCount2, cornerPointSharp, laserCloudCornerLast)
        if len(laserCloudOri) < 10:
            continue
        if calculateTransformationCorner(iterCount2, laserCloudOri, coeffSel, transformCur) == False:
            break
        
def interateTransformation(transformSum, transformCur):
     rx, ry, rz = AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
     -transformCur[0], -transformCur[1], -transformCur[2])

     x1 = cos(rz) * transformCur[3] - sin(rz) * transformCur[4]
     y1 = sin(rz) * transformCur[3] + cos(rz) * transformCur[4]
     z1 = transformCur[5]

     x2 = x1
     y2 = cos(rx) * y1 - sin(rx) * z1
     z2 = sin(rx) * y1 + cos(rx) * z1

     tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2)
     ty = transformSum[4] - y2
     tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2)


     transformSum[0] = rx
     transformSum[1] = ry
     transformSum[2] = rz
     transformSum[3] = tx
     transformSum[4] = ty
     transformSum[5] = tz
