import numpy as np
import socket

import utils

class Controller:
    def __init__(self, port=8888, ip='192.168.123.161', resolution=0.03, lin_speed_range=np.array([0, 1.0]) / 0.03, ang_speed_range=np.array([-0.4, 0.4])):
        self.port = port
        self.ip = ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.resolution = resolution
        self.stop_u = np.array([0, 0])
        self.lin_speed_range = lin_speed_range
        self.ang_speed_range = ang_speed_range
    
    def _send_udp(self, command):
        self.sock.sendto((str(command[0]) + ' ' + str(command[1])).encode(), (self.ip, self.port))
    
    def action_m2w(self, u):
        return np.array([u[0] * self.resolution, u[1]])
    
    def clip_lin_speed(self, speed):
        return np.clip(speed, self.lin_speed_range[0], self.lin_speed_range[1])
    
    def clip_ang_speed(self, speed):
        return np.clip(speed, self.ang_speed_range[0], self.ang_speed_range[1])

    def send_command(self, use_lqr, stop, K, k, x, x_hat, u_hat):
        if stop:
            self._send_udp(self.stop_u)
            return
        
        if use_lqr:
            x_error = np.array([x[0] - x_hat[0], x[1] - x_hat[1], utils.angle_normalize(x[2] - x_hat[2])])[:, np.newaxis]
            u = (K @ x_error).squeeze() + k + u_hat
        else:
            u = u_hat

        u_clipped = np.array([0.0, 0.0])
        u_clipped[0] = self.clip_lin_speed(u[0])
        u_clipped[1] = self.clip_ang_speed(u[1])
        
        u_w = self.action_m2w(u_clipped)
        self._send_udp(u_w)

        return u_clipped
