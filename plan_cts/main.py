from numpy.ma import masked_greater
import rospy
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from fmm import FMMPlanner
from spline import Spline
import utils
from lqr import LQR
from slam import SLAM
from controller import Controller
np.set_printoptions(precision=3, suppress=True)

def main():
    # CAUSTION: 
    #   coord in y, x (reverse order)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action = 'store_true')
    args = parser.parse_args()

    target_pos_w = np.array([0, 2.44])

    slam = SLAM(blur_sigma=0.5)
    print("-----sleep for 4s, building up perception buffer-----")
    time.sleep(4)

    controller = Controller()
    spline = Spline()
    lqr = LQR()

    counter = 1
    # ros_rate_4 = rospy.Rate(4)
    ros_rate_20 = rospy.Rate(20)
    plt.figure(figsize=(10, 5))

    while not rospy.is_shutdown():
        start_time = time.time()
        print('\n--------------------------')

        # map and FMM plan
        t1 = time.time()
        map, _, target_pos, _, _, raw_map = slam.get_global_map(target_pos_w)
        utils.print_time(t1, 'map')
        # curr_vel = np.array([0, 0.01])

        t2 = time.time()
        planner = FMMPlanner(map)
        dd = planner.set_goal(target_pos)
        utils.print_time(t2, 'planner')

        t3 = time.time()
        curr_pos, yaw, curr_vel = slam.get_odom_m()
        next_pos_l, next_vel_l, stop = planner.get_short_term_goal(curr_pos, yaw, np.linalg.norm(curr_vel))
        utils.print_time(t3, 'sample spline')
        
        # spline
        t4 = time.time()
        curr_pos_l = np.stack([curr_pos] * next_pos_l.shape[0])
        curr_vel_l = np.stack([curr_vel] * next_pos_l.shape[0])
        next_pos_trajs = spline.fit_eval(curr_pos_l, next_pos_l, curr_vel_l, next_vel_l)
        utils.print_time(t4, 'fit')

        t4_2 = time.time()
        best_traj_idx = planner.find_argmin_traj(next_pos_trajs)
        utils.print_time(t4_2, 'argmin')

        t5 = time.time()
        next_pos, next_vel = next_pos_l[best_traj_idx], next_vel_l[best_traj_idx]
        xs_raw, us_raw = spline.fit_reference(curr_pos, next_pos, curr_vel, next_vel)
        utils.print_time(t5, 'fit reference')

        # LQR
        use_lqr = False
        us_clipped = []
        if use_lqr: 
            t6 = time.time()
            Ks, ks = lqr.backward_pass(xs_raw, us_raw)
            utils.print_time(t6, 'lqr')
            for t in range(5):
                curr_pos, yaw, _ = slam.get_odom_m()
                x_t = np.array([curr_pos[1], curr_pos[0], yaw])
                x_hat_t = xs_raw[t]
                u_hat_t = us_raw[t]
                u_clipped = controller.send_command(use_lqr, stop, Ks[t], ks[t], x_t, x_hat_t, u_hat_t)
                us_clipped.append(u_clipped)
                if t != 4:
                    ros_rate_20.sleep()
        else:
            for t in range(5):
                u_hat_t = us_raw[t]
                u_clipped = controller.send_command(use_lqr, stop, None, None, None, None, u_hat_t)
                us_clipped.append(u_clipped)
                if t != 4:
                    time.sleep(0.049)
                # if t != 4:
                #     ros_rate_20.sleep()
        
        print('map size: ', map.shape)
        print('curr yaw: ', yaw)
        print('curr position: ', curr_pos)
        print('curr vel: ',  curr_vel)
        print('next position: ', next_pos)
        print('next vel: ', next_vel)
        print('next speed: ', np.linalg.norm(next_vel))
        print('target position: ', target_pos)
        print('us_raw: ', us_raw)
        print('us_clipped: ', np.array(us_clipped))
        print()

        if args.plot:
            # for i in range(len(next_pos_trajs)):
            plt.clf()

            # global plot
            plt.subplot(1, 2, 1)
            plt.imshow(dd, origin='lower')
            next_pos_traj = next_pos_trajs[best_traj_idx]
            plt.plot(next_pos_traj[1:-1, 1] - 0.5, next_pos_traj[1:-1, 0] - 0.5, 'yo', markersize=5)
            plt.arrow(curr_pos[1] - 0.5, curr_pos[0] - 0.5, curr_vel[1], curr_vel[0], color='r', width=0.05)
            plt.arrow(next_pos[1] - 0.5, next_pos[0] - 0.5, next_vel[1], next_vel[0], color='r', width=0.05)
            plt.plot([curr_pos[1] - 0.5], [curr_pos[0] - 0.5], 'ro', markersize=5)
            plt.plot([target_pos[1] - 0.5], [target_pos[0] - 0.5], 'ro', markersize=5)
            plt.plot([next_pos[1] - 0.5], [next_pos[0] - 0.5], 'g+', markersize=10)
            plt.colorbar()

            # local plot
            plt.subplot(1, 2, 2)
            plt.imshow(raw_map, origin='lower')

            # plt.imshow(local_cost, origin='lower', cmap='spring')
            # plt.plot(next_pos[1] - curr_pos[1] + local_cost.shape[1] / 2 - 0.5, next_pos[0] - curr_pos[0] + local_cost.shape[0] / 2 - 0.5, 'yo', markersize=5)
            # plt.colorbar()

            plt.show(block=False)
            plt.pause(1e-6)

        print('time(s) this round: ', (time.time() - start_time))
        
        if stop:
            break
        counter += 1

        # ros_rate_4.sleep()


if __name__ == '__main__':
    main()
