#! /usr/bin/env python

import rospy
import copy
import IPython
import numpy
from franka_interface import ArmInterface 

def gripRecord(gripForce, iteration=0, recordTime=10):
    rospy.init_node('netft_node') #TODO needed?

    zeroFTSensor()
    rospy.sleep(2)

    # Create a folder for the bag
    bagName = 'f{}_i{}'.format(gripForce, iteration)
    dir_save_bagfile = 'gripTests/'
    if not os.path.exists(dir_save_bagfile):
        os.makedirs(dir_save_bagfile)

    topics = ["/netft/netft_data"]
    subprocess.Popen('rosbag record -q -O {} {}'.format(bagName, " ".join(topics)), shell=True, cwd=dir_save_bagfile)   
    rospy.sleep(1)
    #hand.Grasp() TODO
    rospy.sleep(recordTime)
        
    # Stop recording rosbag
    terminate_ros_node("/record")
    rospy.sleep(1)
    #hand.Open() TODO

def zeroFTSensor():
    zeroSensorRos = rospy.ServiceProxy('/netft/zero', srv.Zero)
    rospy.wait_for_service('/netft/zero', timeout=0.5)
    zeroSensorRos()

def terminate_ros_node(s):
    list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for string in list_output.split("\n"):
        if string.startswith(s):
            os.system("rosnode kill " + string)

if __name__ == '__main__':
    rospy.init_node("grip_tests")
    realarm = ArmInterface()
    q0 = realarm.joint_angles()

    IPython.embed()
