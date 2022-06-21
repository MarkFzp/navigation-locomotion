import numpy as np
import matplotlib.pyplot as plt
import utils

class Spline:
    def __init__(self, eval_interval=0.05, control_interval=0.05) -> None:
        self.T_func = lambda ts: np.stack([ts ** 3, ts ** 2, ts, np.full(ts.shape, 1)], axis=1)[np.newaxis, :, np.newaxis, :]
        self.T_dot_func = lambda ts: np.stack([3 * ts ** 2, 2 * ts, np.full(ts.shape, 1), np.full(ts.shape, 0)], axis=1)[np.newaxis, :, np.newaxis, :]
        self.T_ddot_func = lambda ts: np.stack([6 * ts, np.full(ts.shape, 2), np.full(ts.shape, 0), np.full(ts.shape, 0)], axis=1)[np.newaxis, :, np.newaxis, :]
        
        self.ts_eval = np.arange(0, 1.5 + eval_interval, eval_interval)
        self.T_eval = self.T_func(self.ts_eval)
        self.T_dot_eval = self.T_dot_func(self.ts_eval)
        self.T_ddot_eval = self.T_ddot_func(self.ts_eval)

        self.ts_control = np.arange(control_interval, 1 + control_interval, control_interval)
        self.T_control = self.T_func(self.ts_control)
        self.T_dot_control = self.T_dot_func(self.ts_control)
        self.T_ddot_control = self.T_ddot_func(self.ts_control)

        self.M = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
    
    def fit_eval(self, start_pts, end_pts, start_grads, end_grads):
        G = np.stack([start_pts, end_pts, start_grads, end_grads], axis=1) # [N, 4, 2]
        MG = (self.M @ G)[:, np.newaxis, :, :] # [N, T, 4, 2]
        pos = (self.T_eval @ MG).squeeze(axis=2)
        return pos
    
    def fit_reference(self, start_pts, end_pts, start_grads, end_grads):
        G = np.stack([start_pts, end_pts, start_grads, end_grads], axis=0) # [4, 2]
        MG = (self.M @ G)[np.newaxis, np.newaxis, :, :]  # [4, 2]

        # in y, x
        # self.T_control @ MG outputs [1, T, 1, 2]
        pos = (self.T_control @ MG).squeeze() # [T, 2]
        xy_dot = (self.T_dot_control @ MG).squeeze()
        xy_ddot = (self.T_ddot_control @ MG).squeeze()
        
        ang = np.arctan2(xy_dot[:, 0], xy_dot[:, 1])
        lin_speed = np.linalg.norm(xy_dot, ord=2, axis=-1)
        
        y_dot, x_dot = xy_dot[:, 0], xy_dot[:, 1]
        y_ddot, x_ddot = xy_ddot[:, 0], xy_ddot[:, 1]

        ang_speed = (x_dot * y_ddot - y_dot * x_ddot) / lin_speed ** 2

        xs = np.stack([pos[:, 1], pos[:, 0], ang], axis=-1)
        us = np.stack([lin_speed, ang_speed], axis=-1)[: -1]

        return xs, us
