# load lidar data from txt files

import cv2
import numpy as np

def loadLidarData(filename):
    # Input: txt file format: x, y, z, I
    # Output: array lidarData: lidarData[i, :] = [x_i, y_i, z_i, I_i]
    lidarData = []
    with open(filename) as f:
        for line in f:
            data = [int(i) for i in line.split(' ')]
            lidarData.append(data)
    lidarData = np.array(lidarData)
    return lidarData

if __name__ == '__main__':
    lidarData = loadLidarData('test.txt')
    print(lidarData[1,:])