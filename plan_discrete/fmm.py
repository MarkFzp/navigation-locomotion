import numpy as np
import skfmm
import skimage
from numpy import ma


def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    for j in range(size):
        for i in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= step_size ** 2 \
            and ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 > (step_size - 1) ** 2:
                mask[j, i] = 1

    # mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for j in range(size):
        for i in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= step_size ** 2:
                mask[j, i] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversible, scale=1, step_size=5, add_obstacle_dist=True, obstacle_dist_cap=25, obstacle_dist_ratio=1.):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))
        self.fmm_dist = None
        self.add_obstacle_dist = add_obstacle_dist
        self.obstacle_dist_cap = obstacle_dist_cap
        self.obstacle_dist_ratio = obstacle_dist_ratio

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_y, goal_x = int(goal[0] / (self.scale * 1.)), \
            int(goal[1] / (self.scale * 1.))

        if self.traversible[goal_y, goal_x] == 0. and auto_improve:
            goal_y, goal_x = self._find_nearest_goal([goal_y, goal_x])

        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) * 2)

        if self.add_obstacle_dist:
            helper_planner = FMMPlanner(self.traversible, add_obstacle_dist=False)
            obstacle_dist = helper_planner.set_multi_goal(self.traversible == 0)
            inverse_obstacle_dist = np.minimum(self.obstacle_dist_cap, np.power(obstacle_dist, 2))
            dd += self.obstacle_dist_ratio * (self.obstacle_dist_cap - inverse_obstacle_dist)

        self.fmm_dist = dd

        return dd

    def set_multi_goal(self, goal_map):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) * 2)
        self.fmm_dist = dd
        return dd

    def get_short_term_goal(self, state):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dy, dx = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        # dist = np.pad(self.fmm_dist, self.du, 'edge')
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        if subset[self.du, self.du] < 0.25 / 0.03:  # 25cm
            stop = True
        else:
            stop = False

        # subset -= subset[self.du, self.du]
        # ratio1 = subset / dist_mask
        # subset[ratio1 < -1.5] = 1

        (stg_y, stg_x) = np.unravel_index(np.argmin(subset), subset.shape)

        # if subset[stg_y, stg_x] > -0.0001:
        #     replan = True
        # else:
        #     replan = False
        replan = False

        return (stg_y + 0.5 + state[0] - self.du) * scale, \
               (stg_x + 0.5 + state[1] - self.du) * scale, replan, stop, subset

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
