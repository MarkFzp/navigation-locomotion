import time
import socket
# import threading
import multiprocessing
from multiprocessing import Process


class ObstacleDetector:
    def __init__(self, port=9999, ip='192.168.123.12') -> None:
        self.manager = multiprocessing.Manager()
        self.got_hit_flag = multiprocessing.Value('i', 0)
        self.port = port
        self.ip = ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        # self.got_hit_flag = False
        self.udp_thread = Process(target=self._start_receiving, args=(self.got_hit_flag,))
        self.udp_thread.start()

    def hit(self):
        if self.got_hit_flag.value == 1:
            self.got_hit_flag.value = 0
            return True

        return False

    def _start_receiving(self, got_hit_flag):
        while True:
            data, addr = self.sock.recvfrom(64)
            print("[UDP received]: ", data)
            if data and data.decode()[0] == '1':
                print("[set hit flag]")
                got_hit_flag.value = 1


if __name__ == '__main__':
    detector = ObstacleDetector()
    while True:
        time.sleep(1)
        print(detector.hit())
    

