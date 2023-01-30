#! /usr/bin/env python


"""
 @info: 
   commands robot to move to neutral pose

"""

import rospy
import copy
import IPython
import numpy
from franka_interface import ArmInterface 

if __name__ == '__main__':
    rospy.init_node("path_testing")
    arm = ArmInterface()
    frames = arm.get_frames_interface()
    start_pose = arm.endpoint_pose()

    pose0 = copy.deepcopy(start_pose)
    pose1 = copy.deepcopy(pose0)
    pose1['position'][0] += 0.1
    pose2 = copy.deepcopy(pose1)
    pose2['position'][1] += 0.1
    pose3 = copy.deepcopy(pose2)
    pose3['position'][0] -= 0.1
    pose4 = copy.deepcopy(pose3)
    pose4['position'][1] -= 0.1

    path = [pose1, pose2, pose3, pose4]
    

    IPython.embed()
