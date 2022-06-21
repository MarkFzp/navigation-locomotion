import numpy as np
import socket

import utils

class Controller:
    def __init__(self, lin_speed_range, ang_speed_range, in_place_ang_speed_range, port=8888, ip='192.168.123.161', resolution=0.03):
        self.port = port
        self.ip = ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.resolution = resolution
        # self.stop_u = np.array([0, 0])
        self.lin_speed_range = lin_speed_range
        self.ang_speed_range = ang_speed_range
        self.in_place_ang_spee_range = in_place_ang_speed_range
    
    def _send_udp(self, lin_speed, ang_speed):
        udp_msg = str(lin_speed) + ' ' + str(ang_speed)
        print('udp_msg: ', udp_msg)
        self.sock.sendto(udp_msg.encode(), (self.ip, self.port))
    
    def action_m2w(self, lin_speed, ang_speed):
        return lin_speed * self.resolution, ang_speed
    
    def _get_sign(self, number):
        return (number >= 0) * 2 - 1
    
    def clip_and_scale_w(self, lin_speed, ang_speed):
        if lin_speed >= self.lin_speed_range[0]:
            lin_speed_clipped = np.clip(lin_speed, self.lin_speed_range[0], self.lin_speed_range[1]) + 0.15
            ang_speed_clipped = np.clip(ang_speed, self.ang_speed_range[0], self.ang_speed_range[1])
        else:
            lin_speed_clipped = max(lin_speed, 0.15)
            ang_speed_clipped = self._get_sign(ang_speed) * 0.7
        
        return np.array([lin_speed_clipped, ang_speed_clipped])

    def send_command(self, stop, lin_speed, ang_speed):
        if stop:
            self._send_udp(0, 0)
            return

        lin_speed_w, ang_speed_w = self.action_m2w(lin_speed, ang_speed)
        clipped_lin_speed_w, clipped_ang_speed_w = self.clip_and_scale_w(lin_speed_w, ang_speed_w)
        
        self._send_udp(clipped_lin_speed_w, clipped_ang_speed_w)

        return (clipped_lin_speed_w, clipped_ang_speed_w)
