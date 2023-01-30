#! /usr/bin/env python


"""
 @info: 
   commands robot to move to neutral pose

"""

import os
import math
import rospy
import copy
import IPython
import numpy
import random
from franka_interface import ArmInterface 

def randomRealConfiguration(arm):
    limits = arm.get_joint_limits()
    lower = limits.position_lower
    upper = limits.position_upper
    dofs = numpy.zeros(len(lower))
    for i in xrange(len(lower)):
        dofs[i] = random.uniform(lower[i], upper[i])
    return arm.convertToDict(dofs)

def randomQ(arm):
    (lower, upper) = arm.GetJointLimits()
    dofs = numpy.zeros(len(lower)) 
    lower[0] = -math.pi / 2.0
    upper[0] = math.pi / 2.0
    for i in xrange(len(lower)):
        dofs[i] = random.uniform(lower[i], upper[i])
    return dofs

if __name__ == '__main__':
    rospy.init_node("path_testing")
    realarm = ArmInterface()
    q0 = realarm.convertToList(realarm.joint_angles()) 
    startq = realarm.joint_angles()

    while True:
        # Rather than peturbing pose, perturb q as a hack
        delta_q = numpy.random.rand(7)*0.2
        q1_seed = numpy.add(q0, delta_q)

        raw_input("seed")
        realarm.move_to_joint_positions(realarm.convertToDict(q1_seed))
        p1 = realarm.endpoint_pose()
        raw_input("q0")
        realarm.move_to_joint_positions(realarm.convertToDict(q0))
        p0 = realarm.endpoint_pose()

        raw_input("p1")
        realarm.set_cart_impedance_pose(p1)
        raw_input("record?")
        p2 = realarm.endpoint_pose()
        q1 = realarm.joint_angles()
        all_results = [q0, realarm.convertToList(q1), p0, p1, p2] 
        raw_input("next?")
        break

    with open('t28.npy', 'wb') as f:
        numpy.save(f, all_results)

    realarm.set_cart_impedance_pose(p1, stiffness=[0]*6)
    IPython.embed()
