# load lidar data from txt files

import cv2
import numpy as np
from queue import Queue
from matplotlib import cm
import open3d as o3d

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def loadLidarData(filename):
    # Input: txt file format: x, y, z, I
    # Output: array lidarData: lidarData[i, :] = [x_i, y_i, z_i, I_i]
    return np.loadtxt(filename, delimiter=' ')

def visulizeLiadarData(lidarData):
    dataList = []
    for i in range(lidarData.shape[0]):
        if lidarData[i, 3] == 0:
            continue
        dataList.append([v for v in lidarData[i, :]])
    dataList = np.array(dataList)

    # Isolate the intensity and compute a color for it
    intensity = dataList[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
  
    # Isolate the 3D data
    points = dataList[:, :-1]
    
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    vis_points = o3d.utility.Vector3dVector(points)
    vis_colors = o3d.utility.Vector3dVector(int_color)

    point_list = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Carla Lidar',
        width=960,
        height=540,
        left=480,
        top=270)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True

    point_list.points = vis_points
    point_list.colors = vis_colors
    while True:
        try:
            vis.add_geometry(point_list)

            vis.poll_events()
            vis.update_renderer()
        except KeyboardInterrupt:
            print(' - Exited by user.')
            break

    vis.destroy_window()
    
if __name__ == '__main__':
    lidarData = loadLidarData("G:/Carla/WindowsNoEditor/PythonAPI/output/lidar/2023-01-13-21-54-07-520741.txt")
    visulizeLiadarData(lidarData)