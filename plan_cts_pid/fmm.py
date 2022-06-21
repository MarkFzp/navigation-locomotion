import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib import utils
import skfmm
import skimage
from numpy import ma
import cv2
from math import ceil


def get_mask(sx, sy, step_size):
    size = int(step_size) * 2 + 1
    mask = np.zeros((size, size))
    for j in range(size):
        for i in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= step_size ** 2 \
            and ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 > (step_size - 1) ** 2:
                mask[j, i] = 1

    # mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, step_size):
    size = int(step_size) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for j in range(size):
        for i in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= step_size ** 2:
                mask[j, i] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversible, scale, lin_speed_range, resolution=0.03, add_obstacle_dist=True, obstacle_dist_cap=32, obstacle_dist_ratio=0.5, dt=0.1):
        if scale is not None:
            # self.traversible = cv2.resize(traversible, (ceil(traversible.shape[1] / scale) + 1, ceil(traversible.shape[0] / scale) + 1), interpolation=cv2.INTER_LINEAR)
            self.traversible = cv2.resize(traversible, (ceil(traversible.shape[1] / scale) + 2, ceil(traversible.shape[0] / scale) + 2), interpolation=cv2.INTER_LINEAR)
            # self.traversible = cv2.resize(traversible, (ceil(traversible.shape[1] / scale), ceil(traversible.shape[0] / scale)), interpolation=cv2.INTER_LINEAR)
        else:
            self.traversible = traversible
        self.scale = scale
        self.fmm_dist = None
        self.add_obstacle_dist = add_obstacle_dist
        self.obstacle_dist_cap = obstacle_dist_cap / self.scale if scale is not None else obstacle_dist_cap
        self.obstacle_dist_ratio = obstacle_dist_ratio
        self.large_value = 1e10
        self.resolution = resolution
        self.grid_offset = np.array([0.5, 0.5])
        self.dt = dt
        self.lin_speed_range = lin_speed_range / self.scale if scale is not None else lin_speed_range

    def set_goal(self, goal, do_plot, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible, 0)
        goal_y, goal_x = int(goal[0] / self.scale), int(goal[1] / self.scale)

        if self.traversible[goal_y, goal_x] == 0. and auto_improve:
            goal_y, goal_x = self._find_nearest_goal([goal_y, goal_x])

        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) if do_plot else self.large_value)

        if self.add_obstacle_dist:
            helper_planner = FMMPlanner(self.traversible, None, self.lin_speed_range, add_obstacle_dist=False)
            obstacle_dist = helper_planner.set_multi_goal(self.traversible == 0)
            inverse_obstacle_dist = np.minimum(self.obstacle_dist_cap, np.power(obstacle_dist, 1))
            dd += self.obstacle_dist_ratio * (self.obstacle_dist_cap - inverse_obstacle_dist)

        self.fmm_dist = dd

        return dd
    
    @staticmethod
    def old_get_target_yaw(dd, curr_pos): # TODO: optimize
        dd_grad_y, dd_grad_x = np.gradient(dd)
        curr_pos_y_idx, curr_pos_x_idx = int(curr_pos[0]), int(curr_pos[1])
        return np.arctan2(- dd_grad_y[curr_pos_y_idx, curr_pos_x_idx], - dd_grad_x[curr_pos_y_idx, curr_pos_x_idx])

    def compute_grad_from_adjacent(self, f1, f2, f3):
        if f1 is None:
            return f3 - f2
        elif f3 is None:
            return f2 - f1
        else:
            return (f3 - f1) / 2

    def get_target_yaw(self, dd, curr_pos): 
        curr_pos_y_idx, curr_pos_x_idx = int(curr_pos[0] / self.scale), int(curr_pos[1] / self.scale)
        curr_fmm_value = dd[curr_pos_y_idx, curr_pos_x_idx]
        up_fmm_value = dd[curr_pos_y_idx + 1, curr_pos_x_idx] if curr_pos_y_idx + 1 < dd.shape[0] else None
        down_fmm_value = dd[curr_pos_y_idx - 1, curr_pos_x_idx] if curr_pos_y_idx - 1 >= 0 else None
        left_fmm_value = dd[curr_pos_y_idx, curr_pos_x_idx - 1] if curr_pos_x_idx - 1 >= 0 else None
        right_fmm_value = dd[curr_pos_y_idx, curr_pos_x_idx + 1] if curr_pos_x_idx + 1 < dd.shape[1] else None

        dx = self.compute_grad_from_adjacent(left_fmm_value, curr_fmm_value, right_fmm_value)
        dy = self.compute_grad_from_adjacent(down_fmm_value, curr_fmm_value, up_fmm_value)

        return np.arctan2(- dy, - dx)
    
    def get_target_lin_speed(self, dd, curr_pos, yaw):
        curr_pos_0, curr_pos_1 = curr_pos[0] / self.scale, curr_pos[1] / self.scale
        forcast_fmm_values = [dd[int(curr_pos_0 + np.sin(yaw) * i), int(curr_pos_1 + np.cos(yaw) * i)] \
            if 0 <= int(curr_pos_0 + np.sin(yaw) * i) < dd.shape[0] \
            and 0 <= int(curr_pos_1 + np.cos(yaw) * i) < dd.shape[1] \
            else self.large_value for i in range(0, int((self.dt * 10) * self.lin_speed_range[1]))]
        
        print('forcast_fmm: ', forcast_fmm_values)
        
        min_fmm_value = np.inf
        target_distance = -1
        for distance, fmm_value in enumerate(forcast_fmm_values):
            if fmm_value <= min_fmm_value:
                target_distance = distance
                min_fmm_value = fmm_value
            else:
                break
        
        target_lin_speed = (target_distance / (self.dt * 10)) * self.scale
        print(target_distance * self.scale, target_lin_speed)
        return target_lin_speed
    
    def get_stop_condition(self, dd, curr_pos):
        curr_pos_0, curr_pos_1 = curr_pos[0] / self.scale, curr_pos[1] / self.scale
        if dd[int(curr_pos_0), int(curr_pos_1)] < 0.5 / self.resolution / self.scale:  # 50cm
            return True
        return False

    def set_multi_goal(self, goal_map):
        traversible_ma = ma.masked_values(self.traversible, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1) # add_obstacle_dist: outputs a nd array (not masked), with 0 at obstacles
        dd = ma.filled(dd, self.large_value) # add_obstacle_dist: no effect
        self.fmm_dist = dd
        return dd

    def _find_nearest_goal(self, goal):
        traversible = skimage.morphology.binary_dilation(
            np.zeros(self.traversible.shape),
            skimage.morphology.disk(2)) != True
        traversible = traversible * 1.
        planner = FMMPlanner(traversible)
        planner.set_goal(goal)

        mask = self.traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal
