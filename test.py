def findCorrespondingCornerFeatures(iterCount, cornerPointsSharp, laserCloudCornerLast, pointSearchCornerInd1, pointSearchCornerInd2):
    cornerPointsSharpNum = len(cornerPointsSharp)
    nearestFeatureSearchSqDist = 25
    laserCloudOri = []
    coeffSel = []
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
                    if laserCloudCornerLast[j][3] > closestPointScan:
                        if pointSqDis < minPointInd2:
                            minPointSqDis2 = pointSqDis
                            minPointInd2 = j
                pointSearchCornerInd1[i] = closestPointInd
                pointSearchCornerInd2[i] = minPointInd2

            if pointSearchCornerInd2[i] >= 0:
                tripod1 = laserCloudCornerLast[pointSearchCornerInd1[i]]
                tripod2 = laserCloudCornerLast[pointSearchCornerInd2[i]]
                coeff = np.zeros((4,1))

                x0 = pointSel[0]
                y0 = pointSel[1]
                z0 = pointSel[2]
                x1 = tripod1[0]
                y1 = tripod1[1]
                z1 = tripod1[2]
                x2 = tripod2[0]
                y2 = tripod2[1]
                z2 = tripod2[2]

                m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))

                a012 = math.sqrt(m11 * m11  + m22 * m22 + m33 * m33)

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