import glob
import os
import sys
import argparse
import random
import time
from datetime import datetime
import numpy as np
from PIL import Image
from matplotlib import cm
import open3d as o3d
import cv2
from queue import Queue
from queue import Empty
import shutil

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def parser():
    argparser = argparse.ArgumentParser(
        description='CARLA simulation for SLAM')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='window resolution (default: 800x600)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=2,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='500000',
        type=int,
        help='lidar points per second (default: 500000)')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1

    return args

def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)

def lidarDisplayWin():
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

    return vis

def simulation(args):
    # create a respo to save
    outputPath = "_out/"
    outputCameraPath = outputPath + "camera/"
    outputLidarPath = outputPath + "lidar/"
    groundTruthPath = outputPath + "gt/"
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
    os.makedirs(outputCameraPath)
    os.mkdir(outputLidarPath)
    os.mkdir(groundTruthPath)

    # Connect to the server
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    delta = 0.05
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta
    world.apply_settings(settings)

    spectator = world.get_spectator()

    vehicle = None
    camera = None
    lidar = None

    try:
        # Search the desired blueprints
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]

        # Configure the blueprints
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        if args.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))

        # Spawn the blueprints
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[0])
        vehicle.set_autopilot(True)
        traffic_manager.ignore_lights_percentage(vehicle, 100)

        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle)

        # display camera and lidar data
        point_list = o3d.geometry.PointCloud()
        vis = lidarDisplayWin()
        cv2.namedWindow("front_cam", 0)

        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        # The sensor data will be saved in thread-safe Queues
        image_queue = Queue()
        lidar_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))
        
        frame = 0
        
        while True:
            world.tick()
            world_frame = world.get_snapshot().frame

            # set spectator above the vehicle
            spec_transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(spec_transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            assert image_data.frame == lidar_data.frame == world_frame

            try:
                # Get the raw BGRA buffer and convert it to an array of RGB of
                # shape (image_data.height, image_data.width, 3).
                im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
                im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
                im_array = im_array[:, :, :3][:, :, ::-1]
                # save images
                im = Image.fromarray(im_array)
                im.save(outputCameraPath + str(image_data.frame) + ".jpg")

                # Get the lidar data and convert it to a numpy array.
                p_cloud_size = len(lidar_data)
                p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
                p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
                # save lidar data
                np.savetxt(outputLidarPath + str(lidar_data.frame) + '.txt', p_cloud)




                # This (4, 4) matrix transforms the points from lidar space to world space.
                lidar_2_world = np.array(lidar.get_transform().get_matrix())
                # This (4, 4) matrix transforms the points from world to sensor coordinates.
                camera_2_world = np.array(camera.get_transform().get_matrix())
                # This (4, 4) matrix transforms the points from world to sensor coordinates.
                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

                # save ground truth
                np.savetxt(groundTruthPath + str(world_frame) + ".txt", np.r_[lidar_2_world, camera_2_world])

                lidar_2_camera = np.dot(world_2_camera, lidar_2_world)

                ''' test lidar to camera
                if frame == 0:
                    old_transformer = lidar_2_camera
                dis = np.linalg.norm(old_transformer - lidar_2_camera)
                if dis > 0.0001:
                    print(dis)
                '''
                # New we must change from UE4's coordinate system to an "standard"
                # camera coordinate system (the same used by OpenCV):

                # ^ z                       . z
                # |                        /
                # |              to:      +-------> x
                # | . x                   |
                # |/                      |
                # +-------> y             v y

                # This can be achieved by multiplying by the following matrix:
                # [[ 0,  1,  0 ],
                #  [ 0,  0, -1 ],
                #  [ 1,  0,  0 ]]

                # Or, in this case, is the same as swapping:
                # (x, y ,z) -> (y, -z, x)
                # point_in_camera_coords = np.array([
                #     sensor_points[1],
                #     sensor_points[2] * -1,
                #     sensor_points[0]])

                # Finally we can use our K matrix to do the actual 3D -> 2D.
                # points_2d = np.dot(K, point_in_camera_coords)

                # Remember to normalize the x, y values by the 3rd value.


                # Visulization of lidar data
                intensity = p_cloud[:, -1]
                intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
                int_color = np.c_[
                    np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                    np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                    np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

                # Isolate the 3D data
                points = p_cloud[:, :-1]
                
                # We're negating the y to correclty visualize a world that matches
                # what we see in Unreal since Open3D uses a right-handed coordinate system
                points[:, :1] = -points[:, :1]
                vis_points = o3d.utility.Vector3dVector(points)
                vis_colors = o3d.utility.Vector3dVector(int_color)
                point_list.points = vis_points
                point_list.colors = vis_colors
                if frame == 0:
                    vis.add_geometry(point_list)
                vis.update_geometry(point_list)

                vis.poll_events()
                vis.update_renderer()
                # # This can fix Open3D jittering issues:
                time.sleep(0.001)

                # Visulization of camera images
                im_dis = im_array[:, :, ::-1]
                cv2.imshow("front_cam",im_dis)
                cv2.waitKey(1)

                frame += 1

            except Exception as e:
                print(e)
                break

    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
        if lidar:
            lidar.destroy()
        if vehicle:
            vehicle.destroy()

if __name__ == "__main__":
    args = parser()
    simulation(args)