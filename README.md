# [CVPR 2022] Codebase for "[Coupling Vision and Proprioception for Navigation of Legged Robots](https://navigation-locomotion.github.io/)"

install intel realsense SDK at https://github.com/IntelRealSense/librealsense

git clone this directionary to `~/catkin_ws/src`

install by running install.sh

recompile by running recompile.sh

create `occupancy/launch/camera.launch`

```xml
<launch>
<include file="$(find realsense2_camera)/launch/rs_d400_and_t265.launch">
    <arg name="serial_no_camera1" value="TO-BE-FILLED"/> 			<!-- Note: Replace with actual serial number (camera1 default: t265)-->
    <arg name="serial_no_camera2" value="TO-BE-FILLED"/> 			<!-- Note: Replace with actual serial number (camera2 default: d400)-->
</include>
</launch>
```

start the realsense cameras by `roslaunch occupancy occupancy_live_rviz.launch`

start planning by running `python plan_[discrete|cts]/main.py`
