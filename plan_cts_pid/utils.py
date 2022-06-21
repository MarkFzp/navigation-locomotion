import numpy as np
import time 

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

def print_time(prev_time, name):
    print('[time] ' + name + ' : ', time.time() - prev_time)
