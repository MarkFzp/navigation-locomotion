import numpy as np
import matplotlib.pyplot as plt

class Spline:
    def __init__(self) -> None:
        self.T_func = lambda ts: np.array([ts ** 3, ts ** 2, ts, 1]).T
        self.T_dot_func = lambda ts: np.array([3 * ts ** 2, 2 * ts, 1, 0]).T
        self.T_ddot_func = lambda ts: np.array([6 * ts, 2, 0, 0]).T
        self.M = np.arrary([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
    
    def fit(self, start_pt, end_pt, start_grad, end_grad):
        self.G = np.array([start_pt, end_pt, start_grad, end_grad])
    
    def get_point(self, ts):
        return self.T_func(ts) @ self.M @ self.G
    
    def get_speed(self, ts):
        xy_dot = self.T_dot_func(ts) @ self.M @ self.G
        xy_ddot = self.T_ddot_func(ts) @ self.M @ self.G
        
        linear_speed = np.linalg.norm(xy_dot, ord=2, axis=1)
        
        x_dot, y_dot = xy_dot[:, 0], xy_dot[:, 1]
        x_ddot, y_ddot = xy_ddot[:, 0], xy_ddot[:, 1]

        angular_speed = (x_dot * y_ddot - y_dot * x_ddot) / linear_speed

        return linear_speed, angular_speed
    
