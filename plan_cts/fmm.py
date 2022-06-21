import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib import utils
import skfmm
import skimage
from numpy import ma


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
    def __init__(self, traversible, add_obstacle_dist=True, obstacle_dist_cap=20, obstacle_dist_ratio=1.):
        self.traversible = traversible
        self.fmm_dist = None
        self.add_obstacle_dist = add_obstacle_dist
        self.obstacle_dist_cap = obstacle_dist_cap
        self.obstacle_dist_ratio = obstacle_dist_ratio
        self.large_value = 1e10
        self.resolution = 0.03
        self.grid_offset = np.array([0.5, 0.5])
        self.fovs = np.linspace(-0.5, 0.5, num=5)
        self.small_fovs = np.linspace(-0.2, 0.2, num=5)
        self.speeds = np.arange(0.05, 0.95, 0.05) / self.resolution
        self.conservative_step_size_factor = 0.9
        # self.step_sizes = np.array([10, 16, 22, 28]) # 0.3, 0.48, 0.66, 0.84
        # self.vels = self._sample_velocity()

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible, 0)
        goal_y, goal_x = int(goal[0]), int(goal[1])

        if self.traversible[goal_y, goal_x] == 0. and auto_improve:
            goal_y, goal_x = self._find_nearest_goal([goal_y, goal_x])

        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, self.large_value)

        if self.add_obstacle_dist:
            helper_planner = FMMPlanner(self.traversible, add_obstacle_dist=False)
            obstacle_dist = helper_planner.set_multi_goal(self.traversible == 0)
            inverse_obstacle_dist = np.minimum(self.obstacle_dist_cap, np.power(obstacle_dist, 2))
            dd += self.obstacle_dist_ratio * (self.obstacle_dist_cap - inverse_obstacle_dist)

        self.fmm_dist = dd

        return dd

    def set_multi_goal(self, goal_map):
        traversible_ma = ma.masked_values(self.traversible, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1) # add_obstacle_dist: outputs a nd array (not masked), with 0 at obstacles
        dd = ma.filled(dd, self.large_value) # add_obstacle_dist: no effect
        self.fmm_dist = dd
        return dd
    
    # def _sample_velocity(self):
    #     step_size_interval = 6
    #     # [num_step_sizes, num_speeds_per_step_size, num_directions, 2]
    #     speeds = np.stack([self.step_sizes - step_size_interval / 3, self.step_sizes, self.step_sizes + step_size_interval / 3], axis=-1)[..., np.newaxis, np.newaxis] # [4, 3, 1, 1]
    #     angles = np.linspace(np.pi / 36, 2 * np.pi, num=36) - np.pi
    #     unit_vels = np.stack([np.sin(angles), np.cos(angles)], axis=-1)[np.newaxis, np.newaxis, ...] # [1, 1, 36, 2]
    #     vels = (speeds * unit_vels).reshape(4, -1, 2) # [4, 36 * 3, 2]
        
    #     return vels

    def get_short_term_goal(self, pos, yaw, lin_speed):
        pos_int = [int(x) for x in pos]

        if self.fmm_dist[pos_int[0], pos_int[1]] < 0.25 / self.resolution:  # 25cm
            stop = True
        else:
            stop = False

        # step_sizes = (lin_speed + self.speeds) / 2 * self.conservative_step_size_factor
        step_sizes = (lin_speed + np.linspace(np.maximum(lin_speed - 0.4, 0.05), np.minimum(lin_speed + 0.4, 0.95), num=10) / self.resolution) / 2 # [n_step]
        # print('speed: ', lin_speed)
        # print(step_sizes)
        fovs_w = yaw + self.fovs
        fovs_w_1d = fovs_w[np.newaxis, :]
        next_pos = np.stack([np.sin(fovs_w_1d), np.cos(fovs_w_1d)], axis=-1) * step_sizes[:, np.newaxis, np.newaxis] + pos # [n_step, n_fov, 2]
        
        local_vel_angle = fovs_w_1d[..., np.newaxis] + self.small_fovs[np.newaxis, np.newaxis, :] # [1, n_fov, n_small_fov]
        next_vel = np.stack([np.sin(local_vel_angle), np.cos(local_vel_angle)], axis=-1) * step_sizes[:, np.newaxis, np.newaxis, np.newaxis] # [n_step, n_fov, n_small_fov, 2]
        next_pos = np.stack([next_pos] * len(self.small_fovs), axis=2)

        next_pos_l = next_pos.reshape(-1, 2)
        next_vel_l = next_vel.reshape(-1, 2)

        return next_pos_l, next_vel_l, stop
    
    def get_fmm_value(self, coords):
        y, x = coords[:, 0], coords[:, 1]
        y_out = np.logical_or(y < 0, y > self.fmm_dist.shape[0])
        x_out = np.logical_or(x < 0, x > self.fmm_dist.shape[1])
        yx_out = np.logical_or(y_out, x_out)
        out_bools = np.stack([yx_out, yx_out], axis=-1)
        out_coords = np.where(yx_out)

        valid_coords = np.where(out_bools, 0, coords.astype(np.int32))
        valid_y, valid_x = valid_coords[:, 0], valid_coords[:, 1]
        fmm_value = self.fmm_dist[(valid_y, valid_x)]
        fmm_value[out_coords] = self.large_value

        return fmm_value
    
    def find_argmin_traj(self, pos_trajs):
        T = pos_trajs.shape[1]
        flat_pos_trajs = pos_trajs.reshape(-1, 2)
        fmm_values = self.get_fmm_value(flat_pos_trajs)
        traj_cost = fmm_values.reshape(-1, T)
        argmin_idx = np.argmin(np.sum(traj_cost, axis=-1))

        return argmin_idx

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
