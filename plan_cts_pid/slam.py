import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from collections import deque
from math import ceil, floor
import numpy as np
from scipy.ndimage import gaussian_filter
import socket
from scipy.spatial.transform import Rotation

class SLAM:
    def __init__(self, max_buffer_size=5, resolution=0.03, blur_sigma=0.5, smooth_lin_vel_ratio=0.5, port=9999, ip='192.168.123.161', use_gaussian_filter=False, \
        use_config_space=True):
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

        self.smooth_lin_vel_ratio = smooth_lin_vel_ratio
        self.smooth_lin_vel = np.array([0, 0])

        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_size = 7   # 9 = 0.27 / 0.03
        assert(self.robot_size % 2 == 1)
        self.half_robot_size = int(self.robot_size / 2)

        self.use_gaussian_filter = use_gaussian_filter
        self.use_config_space = use_config_space

        self.obstacle_pos_w = None
        self.obstacle_neighbor_offset = np.array([[i, j] for i in [-self.resolution, 0, self.resolution] for j in [-self.resolution, 0, self.resolution]])

    def map_callback(self, data):
        self.map_buffer.append(data)

    def odom_callback(self, data):
        self.odom_buffer.append(data)

        y_lin_vel = data.twist.twist.linear.y
        x_lin_vel = data.twist.twist.linear.x
        curr_lin_vel = np.array([data.twist.twist.linear.y, data.twist.twist.linear.x])
        self.smooth_lin_vel = self.smooth_lin_vel_ratio * curr_lin_vel + (1 - self.smooth_lin_vel_ratio) * self.smooth_lin_vel
       
        ang_speed = data.twist.twist.angular.z

        # self.sock.sendto((str(x_lin_vel) + ' ' + str(y_lin_vel) + ' ' + str(ang_speed)).encode(), (self.ip, self.port))

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
    
    def get_odom(self, get_smooth_lin_vel=True):
        data = self.odom_buffer[-1]
        pos = np.array([data.pose.pose.position.y, data.pose.pose.position.x])
        if get_smooth_lin_vel:
            lin_vel = self.smooth_lin_vel
        else:
            lin_vel = np.array([data.twist.twist.linear.y, data.twist.twist.linear.x])

        yaw = Rotation.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y, \
            data.pose.pose.orientation.z, data.pose.pose.orientation.w]).as_euler('zyx')[0]
        
        return pos, yaw, lin_vel

    def w2m(self, world_frame_pos):
        return (world_frame_pos - self.map_orig_pos_w) / self.resolution
    
    def m2mm(self, map_frame_pos):
        return map_frame_pos - self.m2mm_offset
    
    def preprocess_map(self, map):
        map[map == -1] = 0
        if self.use_gaussian_filter:
            map = gaussian_filter(map, sigma=self.blur_sigma)

        map = (map < 50) # TODO
        # map = (map < 75)
        if self.use_config_space:
            y_size, x_size = map.shape
            padded_map = np.pad(map, self.half_robot_size, 'constant', constant_values=1)
            map = np.min(np.stack([padded_map[y: y + y_size, x: x + x_size] for y in range(self.half_robot_size * 2) for x in range(self.half_robot_size * 2)], axis=-1), axis=-1)

        return map
    
    def add_point_to_map(self, map, point_pos_m, point2_pos_m):
        map_len_y_m, map_len_x_m = map.shape
        point_pos_m_y, point_pos_m_x = point_pos_m

        mmap = map
        point_pos_mm = point_pos_m
        point2_pos_mm = point2_pos_m
        mmap_orig_pos_m = np.zeros([2])
        
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

            mmap_orig_pos_m = np.array([min(0, floor(delta_y)), min(0, floor(delta_x))])
            map_orig_pos_mm = - mmap_orig_pos_m
            
            mmap = np.ones([mmap_len_y_mm, mmap_len_x_mm])
            mmap[map_orig_pos_mm[0]:map_orig_pos_mm[0] + map_len_y_m, map_orig_pos_mm[1]:map_orig_pos_mm[1] + map_len_x_m] = map
            
            point_pos_mm = point_pos_m - mmap_orig_pos_m
            point2_pos_mm = point2_pos_m - mmap_orig_pos_m

            self.m2mm_offset += mmap_orig_pos_m
        
        return mmap, point_pos_mm, point2_pos_mm

    def _add_obstacles_to_map(self, map):
        if self.obstacle_pos_w is not None:
            obstacle_pos_m = self.w2m(np.array(self.obstacle_pos_w)).astype(np.int)
            map[(obstacle_pos_m[:, 0], obstacle_pos_m[:, 1])] = 100

    def get_global_map(self, target_pos_w, hit_obstacle):
        self.m2mm_offset = np.zeros([2])
        map, _ = self.get_map()
        pos_w, yaw, lin_vel_w = self.get_odom()

        if hit_obstacle:
            # TODO
            # new_obstacle_w_center = pos_w + np.array([[0.09], [0.12], [0.15], [0.18], [0.21]]) @ np.array([[np.sin(yaw), np.cos(yaw)]])
            # new_obstacles = new_obstacle_w_center + (np.array([[-0.06], [-0.03], [0], [0.03], [0.06]]) @ np.array([[np.sin(yaw + np.pi / 2), np.cos(yaw + np.pi / 2)]])).reshape([5, 1, 2])
            # new_obstacle_w_center = pos_w + np.array([[0.09], [0.12]]) @ np.array([[np.sin(yaw), np.cos(yaw)]])
            # new_obstacles = new_obstacle_w_center + (np.array([[0], [-0.03], [-0.06], [-0.09]]) @ np.array([[np.sin(yaw + np.pi / 2), np.cos(yaw + np.pi / 2)]])).reshape([-1, 1, 2])
            new_obstacle_w_center = pos_w + np.array([[0.09]]) @ np.array([[np.sin(yaw), np.cos(yaw)]])
            new_obstacles = new_obstacle_w_center + (np.array([[-0.03], [0], [0.03]]) @ np.array([[np.sin(yaw + np.pi / 2), np.cos(yaw + np.pi / 2)]])).reshape([5, 1, 2])
            print('\n\tobstacle position: {}'.format(self.w2m(new_obstacles)))
            if self.obstacle_pos_w is None:
                # self.obstacle_pos_w = np.expand_dims(new_obstacle_w, axis=0)
                self.obstacle_pos_w = new_obstacles.reshape([-1, 2])
            else:
                # self.obstacle_pos_w = np.concatenate([self.obstacle_pos_w, np.expand_dims(new_obstacle_w, axis=0)], axis=0)
                self.obstacle_pos_w = np.concatenate([self.obstacle_pos_w, new_obstacles.reshape([-1, 2])], axis=0)

        self._add_obstacles_to_map(map)
        traversible_map = self.preprocess_map(map)

        target_pos_m = self.w2m(target_pos_w)
        pos_m = self.w2m(pos_w)
        lin_vel_m = lin_vel_w / self.resolution
        mmap, pos_mm, target_pos_mm = self.add_point_to_map(traversible_map, pos_m, target_pos_m)
        mmmap, target_pos_mmm, pos_mmm = self.add_point_to_map(mmap, target_pos_mm, pos_mm)

        return mmmap, pos_mmm, target_pos_mmm, yaw, lin_vel_m

    def get_odom_m(self):
        pos_w, yaw, lin_vel_w = self.get_odom()
        pos_m = self.m2mm(self.w2m(pos_w))
        lin_vel_m = lin_vel_w / self.resolution
        
        return pos_m, yaw, lin_vel_m
