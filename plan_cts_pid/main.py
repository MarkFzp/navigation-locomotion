import rospy
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from fmm import FMMPlanner
import utils
from slam import SLAM
from controller import Controller
from obstacle_detector import ObstacleDetector
from terrain_detector import TerrainDetector

np.set_printoptions(precision=3, suppress=True)

def main():
    # CAUSTION: 
    #   coord in y, x (reverse order)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action = 'store_true')
    parser.add_argument("--obstacle", action = 'store_true')
    parser.add_argument("--maxspeed", action = 'store_true')
    args = parser.parse_args()

    # target_pos_w = np.array([17.5, 30])
    # target_pos_w = np.array([5.5, 14.7]) # np.array([5.5, 11.7])
    # target_pos_w = np.array([11.7, 6.5])
    # target_pos_w = np.array([5.5, 14.7])
    target_pos_w = np.array([11, 6])

    slam = SLAM(blur_sigma=0.5)
    print("-----sleep for 5s, building up perception buffer-----")
    # time.sleep(10) # TODO
    time.sleep(10)

    counter = 0
    dt = 0.1
    ros_rate = rospy.Rate(1 / dt)
    plt.figure(figsize=(5, 5))
    yaw_Kp = 1
    yaw_Kd = 0.02
    # lin_speed_Kp = 1
    # lin_speed_Kd = 0.02
    resolution = 0.03
    stop = False
    speed_up_smoother = 0.3
    speed_down_smoother = 0.7
    # scale = 3 #TODO
    scale = 2
    hit_first_obstacle = False

    controller_lin_speed_range_w = np.array([0.25, 0.85]) # np.array([0.25, 0.6])
    controller_ang_speed_range_w = np.array([-0.4, 0.4])
    controller_in_place_ang_speed_range_w = None #np.array([0.4, 0.6])
    planner_lin_speed_range = np.array([0.0, 0.85]) / resolution

    controller = Controller(controller_lin_speed_range_w, controller_ang_speed_range_w, controller_in_place_ang_speed_range_w)
    if args.obstacle:
        obstacle_detector = ObstacleDetector()
    
    if args.maxspeed:
        terrain_detector = TerrainDetector()

    got_hit = False

    prev_yaw = 0
    prev_lin_speed = 0
    prev_cmd_lin_speed = 0
    while not rospy.is_shutdown():
        start_time = time.time()
        print('\n--------------------------')

        # invisible obstacle ##############################
        if args.obstacle:
            t4 = time.time()
            got_hit = obstacle_detector.hit()

            utils.print_time(t4, 'obstacle')
        
        # maxspeed ##############################
        if args.maxspeed:
            t5 = time.time()
            decrease_max_speed = terrain_detector.decrease()
            if decrease_max_speed:
                print('[decrease speed]!')
                controller_lin_speed_range_w[1] = max(0.3, controller_lin_speed_range_w[1] - 0.4)
            else:
                controller_lin_speed_range_w[1] = min(0.8, controller_lin_speed_range_w[1] + 0.2)

            utils.print_time(t5, 'maxspeed')

        # map ##############################
        t1 = time.time()
        map, curr_pos, target_pos, _, _ = slam.get_global_map(target_pos_w, got_hit)
        if counter % 10 == 0  and counter <= 2000:
            np.savez('maps/log_{}.npz'.format(counter), map=map, target=target_pos, current=curr_pos)
        utils.print_time(t1, 'map')

        # FMM plan ##############################
        t2 = time.time()
        planner = FMMPlanner(map.astype(np.float32), scale, planner_lin_speed_range)
        dd = planner.set_goal(target_pos, args.plot)
        
        curr_pos, yaw, curr_vel = slam.get_odom_m()
        if planner.get_stop_condition(dd, curr_pos): #TODO
            stop = True
        # if planner.get_stop_condition(dd, curr_pos):
        #     if not hit_first_obstacle:
        #         hit_first_obstacle = True
        #         target_pos_w = np.array([0, 0])
        #         u_w = controller.send_command(stop, None, None)
        #         time.sleep(5)
        #     else:
        #         stop = True
        utils.print_time(t2, 'planner')

        # pid ##############################
        t3 = time.time()
        curr_lin_speed = np.linalg.norm(curr_vel)
        target_yaw = planner.get_target_yaw(dd, curr_pos)
        target_lin_speed = planner.get_target_lin_speed(dd, curr_pos, yaw)

        p_yaw = utils.angle_normalize(target_yaw - yaw)
        d_yaw = 0 - utils.angle_normalize(yaw - prev_yaw) / dt
        # p_lin_speed = target_lin_speed - curr_lin_speed
        # d_lin_speed = 0 - (curr_lin_speed - prev_lin_speed) / dt

        yaw_cmd = yaw_Kp * p_yaw + yaw_Kd * d_yaw
        # lin_speed_cmd = lin_speed_Kp * p_lin_speed + lin_speed_Kd * d_lin_speed
        if target_lin_speed >= curr_lin_speed:
            lin_speed_cmd = speed_up_smoother * target_lin_speed + (1 - speed_up_smoother) * prev_cmd_lin_speed
        else:
            lin_speed_cmd = speed_down_smoother * target_lin_speed + (1 - speed_down_smoother) * prev_cmd_lin_speed
        u_w = controller.send_command(stop, lin_speed_cmd, yaw_cmd)

        prev_yaw = yaw
        # prev_lin_speed = curr_lin_speed
        prev_cmd_lin_speed = lin_speed_cmd
        utils.print_time(t3, 'pid')

        ############################################################
        print('counter: ', counter)
        print('got hit: ', got_hit)
        print('map size: ', map.shape)
        print('curr yaw: ', yaw)
        print('curr position: ', curr_pos)
        print('curr vel: ',  curr_vel)
        print('target speed: ', target_lin_speed)
        print('target yaw: ', target_yaw)
        print('p_yaw: ', p_yaw)
        print('d_yaw: ', d_yaw)
        # print('p_lin_speed: ', p_lin_speed)
        # print('d_lin_speed: ', d_lin_speed)
        print('yaw_cmd: ', yaw_cmd)
        print('lin_speed_cmd: ', lin_speed_cmd)
        print('u_w: ', u_w)
        print('stop: ', stop)
        print('max_speed: ', controller_lin_speed_range_w[1])
        print()

        if args.plot:
            plt.clf()

            # global plot
            # plt.subplot(1, 2, 1)
            plt.imshow(dd, origin='lower')
            plt.arrow((curr_pos[1] / scale) - 0.5, (curr_pos[0] / scale) - 0.5, curr_vel[1] / scale, curr_vel[0] / scale, color='r', width=0.05)
            plt.arrow((curr_pos[1] / scale) - 0.5, (curr_pos[0] / scale) - 0.5, np.cos(target_yaw) * target_lin_speed / scale, np.sin(target_yaw) * target_lin_speed / scale, color='y', width=0.05)
            plt.plot([(curr_pos[1] / scale) - 0.5], [(curr_pos[0] / scale) - 0.5], 'ro', markersize=5)
            plt.plot([(target_pos[1] / scale) - 0.5], [(target_pos[0] / scale) - 0.5], 'ro', markersize=5)
            plt.colorbar()
            # plt.savefig('map_{}.png'.format(counter))

            # local plot
            # plt.subplot(1, 2, 2)
            # plt.imshow(local_cost, origin='lower', cmap='spring')
            # plt.plot(next_pos[1] - curr_pos[1] + local_cost.shape[1] / 2 - 0.5, next_pos[0] - curr_pos[0] + local_cost.shape[0] / 2 - 0.5, 'yo', markersize=5)
            # plt.colorbar()

            plt.show(block=False)
            plt.pause(1e-6)

        
        if stop: #TODO
            break
        counter += 1

        ros_rate.sleep()
        utils.print_time(start_time, 'time(s) this round: ')


if __name__ == '__main__':
    main()
