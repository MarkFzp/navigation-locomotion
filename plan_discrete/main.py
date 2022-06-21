from numpy.ma import common_fill_value
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry

from collections import deque
from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import socket
import time
from scipy.spatial.transform import Rotation
import math
import time
import argparse

from fmm import FMMPlanner

resolution = None

class SLAM:
    def __init__(self, max_buffer_size=5, resolution=0.03, blur_sigma=1.5):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('occupancy', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/t265/odom/sample', Odometry, self.odom_callback)

        self.max_buffer_size = max_buffer_size
        self.resolution = resolution
        self.blur_sigma = blur_sigma

        self.map_buffer = deque(maxlen=max_buffer_size)
        self.odom_buffer = deque(maxlen=max_buffer_size)

        self.map_orig_pos_w = None
        self.mmap_orig_pos_m = None

    def map_callback(self, data):
        self.map_buffer.append(data)

    def odom_callback(self, data):
        self.odom_buffer.append(data)

    def get_map(self):
        data = self.map_buffer[-1]
        map = data.data
        map_ref_pt = np.array([data.info.origin.position.y, data.info.origin.position.x])
        map_raw = np.array(map).astype(dtype=np.int8).reshape([data.info.height, data.info.width])

        map = (map_raw.T)[::-1, :]
        map_size = np.array([data.info.width, data.info.height])
        # print(map.shape)
        # plt.imshow(map, origin='lower')
        # plt.show()
        
        map_size_w = map_size * self.resolution
        self.map_orig_pos_w = np.array([map_ref_pt[0] - map_size_w[0], map_ref_pt[1]])

        return map, self.map_orig_pos_w
    
    def get_odom(self):
        data = self.odom_buffer[-1]
        pos = np.array([data.pose.pose.position.y, data.pose.pose.position.x])
        yaw = Rotation.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y, \
            data.pose.pose.orientation.z, data.pose.pose.orientation.w]).as_euler('zyx', degrees=True)[0]
        return pos, yaw

    def w2m(self, world_frame_pos):
        return (world_frame_pos - self.map_orig_pos_w) / self.resolution
    
    # def m2mm(self, map_frame_pos):
    #     return map_frame_pos - self.mmap_orig_pos_m
    
    def preprocess_map(self, map):
        map[map == -1] = 0
        map = gaussian_filter(map, sigma=self.blur_sigma)
        map = (map < 25)

        return map
    
    def add_point_to_map(self, map, point_pos_m):
        map_len_y_m, map_len_x_m = map.shape
        point_pos_m_y, point_pos_m_x = point_pos_m

        mmap = map
        # pos_mm = pos_m
        point_pos_mm = point_pos_m
        transform_fn = lambda x: x
        self.mmap_orig_pos_m = np.zeros([2])
        
        if not ((0 < point_pos_m_x < map_len_x_m) and (0 < point_pos_m_y < map_len_y_m)):
            delta_x = 0
            delta_y = 0
            
            if point_pos_m_x < 0:
                delta_x = point_pos_m_x
            elif point_pos_m_x > map_len_x_m:
                delta_x = point_pos_m_x - map_len_x_m
            
            if point_pos_m_y < 0:
                delta_y = point_pos_m_y
            elif point_pos_m_y > map_len_y_m:
                delta_y = point_pos_m_y - map_len_y_m
            
            mmap_len_x_mm = map_len_x_m + ceil(abs(delta_x))
            mmap_len_y_mm = map_len_y_m + ceil(abs(delta_y))

            self.mmap_orig_pos_m = np.array([min(0, floor(delta_y)), min(0, floor(delta_x))])
            self.map_orig_pos_mm = - self.mmap_orig_pos_m
            
            mmap = np.ones([mmap_len_y_mm, mmap_len_x_mm])
            mmap[self.map_orig_pos_mm[0]:self.map_orig_pos_mm[0] + map_len_y_m, self.map_orig_pos_mm[1]:self.map_orig_pos_mm[1] + map_len_x_m] = map
            
            transform_fn = lambda map_frame_pos: map_frame_pos - self.mmap_orig_pos_m
            point_pos_mm = transform_fn(point_pos_m)
            # pos_mm = transform_fn(pos_m)
        
        return mmap, point_pos_mm, transform_fn

    def get_global_map(self, target_pos_w):
        map, _ = self.get_map()
        pos_w, yaw = self.get_odom()
        map = self.preprocess_map(map)

        target_pos_m = self.w2m(target_pos_w)
        pos_m = self.w2m(pos_w)

        mmap, pos_mm, transform_fn = self.add_point_to_map(map, pos_m)
        target_pos_mm = transform_fn(target_pos_m)

        mmmap, target_pos_mmm, transform_fn = self.add_point_to_map(mmap, target_pos_mm)
        pos_mmm = transform_fn(pos_mm)

        return mmmap, pos_mmm, yaw, target_pos_mmm


class Controller:
    def __init__(self, port=8888, ip='192.168.123.161', turn_angle=20, unit_time=0.5):
        self.port = port
        self.ip = ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.turn_angle = turn_angle
        self.unit_time = unit_time
    
    def send_udp(self, command):
        self.sock.sendto(str(command).encode(), (self.ip, self.port))
        # time.sleep(self.unit_time)

    def goal2command(self, stop, goal, pos, yaw):
        if stop:
            command = 1
            return command

        target_angle = math.degrees(math.atan2(goal[0] - pos[0], goal[1] - pos[1]))
        print('target_angle: ', target_angle)
        relative_angle = (target_angle - yaw) % 360
        if relative_angle > 180:
            relative_angle -= 360
        print('target_angle: ', relative_angle)
        
        if relative_angle > self.turn_angle:
            command = 2
        elif relative_angle < -self.turn_angle:
            command = 3
        else:
            command = 0
        
        return command
        

def main():
    # CAUSTION: 
    #   coord in y, x (reverse order)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action = 'store_true')
    args = parser.parse_args()

    target_pos = np.array([0, 2.44])

    slam = SLAM(blur_sigma=0.5)
    print("-----sleep for 2s, building up perception buffer-----")
    time.sleep(2)

    controller = Controller()

    start_time = time.time()
    counter = 1

    ros_rate = rospy.Rate(2)

    while not rospy.is_shutdown():
        plt.clf()
        print('\n--------------------------')
        mmap, pos_mm, yaw, target_pos_mm = slam.get_global_map(target_pos)
        planner = FMMPlanner(mmap)
        dd = planner.set_goal(target_pos_mm)
        delta_y, delta_x, replan, stop, local_cost = planner.get_short_term_goal(pos_mm)
        next_pos = np.array([delta_y, delta_x])
        
        print('map size: ', mmap.shape)
        print('robot yaw: ', yaw)
        print('robot position: ', pos_mm)
        print('next position: ', next_pos)
        print('target position: ', target_pos_mm)

        if args.plot:
            plt.subplot(1, 2, 1)
            plt.imshow(dd, origin='lower')
            plt.plot([pos_mm[1]], [pos_mm[0]], 'ro', markersize=5)
            plt.plot([target_pos_mm[1]], [target_pos_mm[0]], 'ro', markersize=5)
            plt.plot([next_pos[1]], [next_pos[0]], 'g+', markersize=10)
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(local_cost, origin='lower', cmap='spring')
            plt.plot([pos_mm[1]- int(pos_mm[1]) + 5], [pos_mm[0]- int(pos_mm[0]) + 5], 'ro', markersize=5)
            plt.colorbar()

            plt.show(block=False)
            plt.pause(0.0001)

        command = controller.goal2command(stop, next_pos, pos_mm, yaw)
        print('command: ', command)
        controller.send_udp(command)

        print('time(s) per round: ', (time.time() - start_time) / counter)
        counter += 1

        ros_rate.sleep()


if __name__ == '__main__':
    main()
