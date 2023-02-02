# /***************************************************************************

#
# @package: franka_interface
# @metapackage: franka_ros_interface
# @author: Saif Sidhik <sxs1412@bham.ac.uk>
#

# **************************************************************************/

# /***************************************************************************
# Copyright (c) 2019, Saif Sidhik

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# **************************************************************************/

"""
 @info:
   Inteface Class for Franka robot arm.

"""


import enum
import rospy
import warnings
import quaternion
import numpy as np
from copy import deepcopy
from rospy_message_converter import message_converter

from franka_core_msgs.msg import (
    JointCommand,
    RobotState,
    EndPointState,
    CartImpedanceStiffness,
    JointImpedanceStiffness,
    TorqueCmd,
    JICmd,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, Wrench
from moveit_msgs.msg import RobotTrajectory

import franka_msgs
import franka_interface
import franka_dataflow
from .robot_params import RobotParams

from franka_tools import (
    FrankaFramesInterface,
    FrankaControllerManagerInterface,
    JointTrajectoryActionClient,
    CollisionBehaviourInterface,
)
import IPython
from pydrake.all import PathParameterizedTrajectory
from typing import Iterable, Optional


def convert_dict_to_joint(joints, joint_names):
    return np.array([joints[name] for name in joint_names])


def convert_dict_to_wrench(wrench):
    return np.concatenate((wrench["force"], wrench["torque"]))


class TipState:
    def __init__(self, timestamp, pose, vel, O_effort, K_effort):
        self.timestamp = timestamp
        self._pose = pose
        self._velocity = vel
        self._effort = O_effort
        self._effort_in_K_frame = K_effort

    @property
    def pose(self):
        return self._pose

    @property
    def velocity(self):
        return self._velocity

    @property
    def effort(self):
        return self._effort

    @property
    def effort_in_K_frame(self):
        return self._effort_in_K_frame


class ArmInterface(object):

    """
    Interface Class for an arm of Franka Panda robot
    Constructor.

    :type synchronous_pub: bool
    :param synchronous_pub: designates the JointCommand Publisher
        as Synchronous if True and Asynchronous if False.

        Synchronous Publishing means that all joint_commands publishing to
        the robot's joints will block until the message has been serialized
        into a buffer and that buffer has been written to the transport
        of every current Subscriber. This yields predicable and consistent
        timing of messages being delivered from this Publisher. However,
        when using this mode, it is possible for a blocking Subscriber to
        prevent the joint_command functions from exiting. Unless you need exact
        JointCommand timing, default to Asynchronous Publishing (False).
    """

    # Containers
    @enum.unique
    class RobotMode(enum.IntEnum):
        """
        Enum class for specifying and retrieving the current robot mode.
        """

        # ----- access using parameters name or value
        # ----- eg. RobotMode(0).name & RobotMode(0).value
        # ----- or  RobotMode['ROBOT_MODE_OTHER'].name & RobotMode['ROBOT_MODE_OTHER'].value

        ROBOT_MODE_OTHER = 0
        ROBOT_MODE_IDLE = 1
        ROBOT_MODE_MOVE = 2
        ROBOT_MODE_GUIDING = 3
        ROBOT_MODE_REFLEX = 4
        ROBOT_MODE_USER_STOPPED = 5
        ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY = 6

    def __init__(self, synchronous_pub=False):
        """ """
        self.hand = franka_interface.GripperInterface()

        self._params = RobotParams()

        self._ns = self._params.get_base_namespace()

        self._joint_limits = self._params.get_joint_limits()

        joint_names = self._joint_limits.joint_names
        if not joint_names:
            rospy.logerr("Cannot detect joint names for arm on this " "robot. Exiting Arm.init().")

            return

        self._joint_names = joint_names
        self.name = self._params.get_robot_name()
        self._joint_angle = dict()
        self._joint_velocity = dict()
        self._joint_effort = dict()
        self._cartesian_pose = dict()
        self._cartesian_velocity = dict()
        self._cartesian_effort = dict()
        self._stiffness_frame_effort = dict()
        self._errors = dict()
        self._collision_state = False
        self._tip_states = None
        self._jacobian = None
        self._cartesian_contact = None

        self._robot_mode = False

        self._command_msg = JointCommand()

        # neutral pose joint positions
        self._neutral_pose_joints = self._params.get_neutral_pose()

        self._frames_interface = FrankaFramesInterface()

        try:
            self._collision_behaviour_interface = CollisionBehaviourInterface()
        except rospy.ROSException:
            rospy.loginfo(
                "Collision Service Not found. It will not be possible to change collision behaviour of robot!"
            )
            self._collision_behaviour_interface = None
        self._ctrl_manager = FrankaControllerManagerInterface(ns=self._ns, sim=self._params._in_sim)

        self._speed_ratio = 0.15

        queue_size = None if synchronous_pub else 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._joint_command_publisher = rospy.Publisher(
                self._ns + "/motion_controller/arm/joint_commands",
                JointCommand,
                tcp_nodelay=True,
                queue_size=queue_size,
            )

        self._pub_joint_cmd_timeout = rospy.Publisher(
            self._ns + "/motion_controller/arm/joint_command_timeout",
            Float64,
            latch=True,
            queue_size=10,
        )

        self._robot_state_subscriber = rospy.Subscriber(
            self._ns + "/custom_franka_state_controller/robot_state",
            RobotState,
            self._on_robot_state,
            queue_size=1,
            tcp_nodelay=True,
        )

        joint_state_topic = self._ns + "/custom_franka_state_controller/joint_states"
        self._joint_state_sub = rospy.Subscriber(
            joint_state_topic,
            JointState,
            self._on_joint_states,
            queue_size=1,
            tcp_nodelay=True,
        )

        self._cartesian_state_sub = rospy.Subscriber(
            self._ns + "/custom_franka_state_controller/tip_state",
            EndPointState,
            self._on_endpoint_state,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Cartesian Impedance Controller Publishers
        self._cartesian_impedance_pose_publisher = rospy.Publisher("equilibrium_pose", PoseStamped, queue_size=10)
        self._cartesian_stiffness_publisher = rospy.Publisher(
            "impedance_stiffness", CartImpedanceStiffness, queue_size=10
        )

        # Force Control Publisher
        self._force_controller_publisher = rospy.Publisher("wrench_target", Wrench, queue_size=10)

        # Torque Control Publisher
        self._torque_controller_publisher = rospy.Publisher("torque_target", TorqueCmd, queue_size=20)

        # Joint Impedance Controller Publishers
        self._joint_impedance_publisher = rospy.Publisher("joint_impedance_position_velocity", JICmd, queue_size=20)
        self._joint_stiffness_publisher = rospy.Publisher(
            "joint_impedance_stiffness", JointImpedanceStiffness, queue_size=10
        )

        rospy.on_shutdown(self._clean_shutdown)

        err_msg = ("%s arm init failed to get current joint_states " "from %s") % (
            self.name.capitalize(),
            joint_state_topic,
        )
        franka_dataflow.wait_for(lambda: len(self._joint_angle.keys()) > 0, timeout_msg=err_msg, timeout=5.0)

        err_msg = ("%s arm, init failed to get current tip_state " "from %s") % (
            self.name.capitalize(),
            self._ns + "tip_state",
        )
        franka_dataflow.wait_for(
            lambda: len(self._cartesian_pose.keys()) > 0,
            timeout_msg=err_msg,
            timeout=5.0,
        )

        err_msg = ("%s arm, init failed to get current robot_state " "from %s") % (
            self.name.capitalize(),
            self._ns + "robot_state",
        )
        franka_dataflow.wait_for(lambda: self._jacobian is not None, timeout_msg=err_msg, timeout=5.0)

        self.set_joint_position_speed(self._speed_ratio)

    def convertToDict(self, q):
        q_dict = dict()
        for i in range(len(q)):
            q_dict["panda_joint{}".format(i + 1)] = q[i]
        return q_dict

    def convertToList(self, q_dict):
        q = []
        sorted_keys = sorted(q_dict.keys())
        for i in sorted_keys:
            q.append(q_dict[i])
        return q

    def _clean_shutdown(self):
        self._joint_state_sub.unregister()
        self._cartesian_state_sub.unregister()
        self._pub_joint_cmd_timeout.unregister()
        self._robot_state_subscriber.unregister()
        self._joint_command_publisher.unregister()
        self._cartesian_impedance_pose_publisher.unregister()
        self._cartesian_stiffness_publisher.unregister()
        self._force_controller_publisher.unregister()
        self._torque_controller_publisher.unregister()
        self._joint_impedance_publisher.unregister()
        self._joint_stiffness_publisher.unregister()

    def get_robot_params(self):
        """
        :return: Useful parameters from the ROS parameter server.
        :rtype: franka_interface.RobotParams
        """
        return self._params

    def get_joint_limits(self):
        """
        Return the joint limits (defined in the parameter server)

        :rtype: franka_core_msgs.msg.JointLimits
        :return: JointLimits
        """
        return self._joint_limits

    def joint_names(self):
        """
        Return the names of the joints for the specified limb.

        :rtype: [str]
        :return: ordered list of joint names from proximal to distal (i.e. shoulder to wrist).
        """
        return self._joint_names

    def _on_joint_states(self, msg):

        for idx, name in enumerate(msg.name):
            if name in self._joint_names:
                self._joint_angle[name] = msg.position[idx]
                self._joint_velocity[name] = msg.velocity[idx]
                self._joint_effort[name] = msg.effort[idx]

    def _on_robot_state(self, msg):

        self._robot_mode = self.RobotMode(msg.robot_mode)

        self._robot_mode_ok = (self._robot_mode.value != self.RobotMode.ROBOT_MODE_REFLEX) and (
            self._robot_mode.value != self.RobotMode.ROBOT_MODE_USER_STOPPED
        )

        self._jacobian = np.asarray(msg.O_Jac_EE).reshape(6, 7, order="F")

        self._cartesian_velocity = {
            "linear": np.asarray([msg.O_dP_EE[0], msg.O_dP_EE[1], msg.O_dP_EE[2]]),
            "angular": np.asarray([msg.O_dP_EE[3], msg.O_dP_EE[4], msg.O_dP_EE[5]]),
        }

        self._cartesian_contact = msg.cartesian_contact
        self._cartesian_collision = msg.cartesian_collision

        self._joint_contact = msg.joint_contact
        self._joint_collision = msg.joint_collision
        if self._frames_interface:
            self._frames_interface._update_frame_data(msg.F_T_EE, msg.EE_T_K)

        self._joint_inertia = np.asarray(msg.mass_matrix).reshape(7, 7, order="F")
        self._external_torque_filtered = np.asarray(msg.tau_ext_hat_filtered)

        self.q_d = msg.q_d
        self.dq_d = msg.dq_d

        self._gravity = np.asarray(msg.gravity)
        self._coriolis = np.asarray(msg.coriolis)

        self._errors = message_converter.convert_ros_message_to_dictionary(msg.current_errors)

    def reactivate(self):
        """
        use automatic recovery
        """
        if not self._robot_mode_ok:
            robot_enable = franka_interface.RobotEnable()
            robot_enable.enable()

    def coriolis_comp(self):
        """
        Return coriolis compensation torques. Useful for compensating coriolis when
        performing direct torque control of the robot.

        :rtype: np.ndarray
        :return: 7D joint torques compensating for coriolis.
        """
        return self._coriolis

    def gravity_comp(self):
        """
        Return gravity compensation torques.

        :rtype: np.ndarray
        :return: 7D joint torques compensating for gravity.
        """
        return self._gravity

    def get_robot_status(self):
        """
        Return dict with all robot status information.

        :rtype: dict
        :return: ['robot_mode' (RobotMode object), 'robot_status' (bool), 'errors' (dict() of errors and their truth value), 'error_in_curr_status' (bool)]
        """
        return {
            "robot_mode": self._robot_mode,
            "robot_status": self._robot_mode_ok,
            "errors": self._errors,
            "error_in_current_state": self.error_in_current_state(),
        }

    def in_safe_state(self):
        """
        Return True if the specified limb is in safe state (no collision, reflex, errors etc.).

        :rtype: bool
        :return: True if the arm is in safe state, False otherwise.
        """
        return self._robot_mode_ok and not self.error_in_current_state()

    def error_in_current_state(self):
        """
        Return True if the specified limb has experienced an error.

        :rtype: bool
        :return: True if the arm has error, False otherwise.
        """
        return not all([e == False for e in self._errors.values()])

    def what_errors(self):
        """
        Return list of error messages if there is error in robot state

        :rtype: [str]
        :return: list of names of current errors in robot state
        """
        return [e for e in self._errors if self._errors[e] == True] if self.error_in_current_state() else None

    def get_external_torque(self):
        return self._external_torque_filtered

    def _on_endpoint_state(self, msg):

        cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4, 4, order="F")
        self.end_effector_pose = cart_pose_trans_mat

        self._cartesian_pose = {
            "position": cart_pose_trans_mat[:3, 3],
            "orientation": quaternion.from_rotation_matrix(cart_pose_trans_mat[:3, :3]),
        }
        self._cartesian_effort = {
            "force": np.asarray(
                [
                    msg.O_F_ext_hat_K.wrench.force.x,
                    msg.O_F_ext_hat_K.wrench.force.y,
                    msg.O_F_ext_hat_K.wrench.force.z,
                ]
            ),
            "torque": np.asarray(
                [
                    msg.O_F_ext_hat_K.wrench.torque.x,
                    msg.O_F_ext_hat_K.wrench.torque.y,
                    msg.O_F_ext_hat_K.wrench.torque.z,
                ]
            ),
        }

        self._stiffness_frame_effort = {
            "force": np.asarray(
                [
                    msg.K_F_ext_hat_K.wrench.force.x,
                    msg.K_F_ext_hat_K.wrench.force.y,
                    msg.K_F_ext_hat_K.wrench.force.z,
                ]
            ),
            "torque": np.asarray(
                [
                    msg.K_F_ext_hat_K.wrench.torque.x,
                    msg.K_F_ext_hat_K.wrench.torque.y,
                    msg.K_F_ext_hat_K.wrench.torque.z,
                ]
            ),
        }

        self._tip_states = TipState(
            msg.header.stamp,
            deepcopy(self._cartesian_pose),
            deepcopy(self._cartesian_velocity),
            deepcopy(self._cartesian_effort),
            deepcopy(self._stiffness_frame_effort),
        )

    def joint_angle(self, joint):
        """
        Return the requested joint angle.

        :type joint: str
        :param joint: name of a joint
        :rtype: float
        :return: angle in radians of individual joint
        """
        return self._joint_angle[joint]

    def joint_angles(self):
        """
        Return all joint angles.

        :rtype: dict({str:float})
        :return: unordered dict of joint name Keys to angle (rad) Values
        """
        return deepcopy(self._joint_angle)

    def joint_ordered_angles(self):
        """
        Return all joint angles.

        :rtype: [float]
        :return: joint angles (rad) orded by joint_names from proximal to distal (i.e. shoulder to wrist).
        """
        return [self._joint_angle[name] for name in self._joint_names]

    def joint_velocity(self, joint):
        """
        Return the requested joint velocity.

        :type joint: str
        :param joint: name of a joint
        :rtype: float
        :return: velocity in radians/s of individual joint
        """
        return self._joint_velocity[joint]

    def joint_velocities(self):
        """
        Return all joint velocities.

        :rtype: dict({str:float})
        :return: unordered dict of joint name Keys to velocity (rad/s) Values
        """
        return deepcopy(self._joint_velocity)

    def joint_effort(self, joint):
        """
        Return the requested joint effort.

        :type joint: str
        :param joint: name of a joint
        :rtype: float
        :return: effort in Nm of individual joint
        """
        return self._joint_effort[joint]

    def joint_efforts(self):
        """
        Return all joint efforts.

        :rtype: dict({str:float})
        :return: unordered dict of joint name Keys to effort (Nm) Values
        """
        return deepcopy(self._joint_effort)

    def endpoint_pose(self):
        """
        Return Cartesian endpoint pose {position, orientation}.

        :rtype: dict({str:np.ndarray (shape:(3,)), str:quaternion.quaternion})
        :return: position and orientation as named tuples in a dict

          - 'position': np.array of x, y, z
          - 'orientation': quaternion x,y,z,w in quaternion format

        """
        return deepcopy(self._cartesian_pose)

    def endpoint_velocity(self):
        """
        Return Cartesian endpoint twist {linear, angular}.

        :rtype: dict({str:np.ndarray (shape:(3,)),str:np.ndarray (shape:(3,))})
        :return: linear and angular velocities as named tuples in a dict

          - 'linear': np.array of x, y, z
          - 'angular': np.array of x, y, z (angular velocity along the axes)
        """
        return deepcopy(self._cartesian_velocity)

    def endpoint_effort(self, in_base_frame=True):
        """
        Return Cartesian endpoint wrench {force, torque}.

        :param in_base_frame: if True, returns end-effector effort with respect to base frame, else in stiffness frame [default: True]
        :type in_base_frame: bool
        :rtype: dict({str:np.ndarray (shape:(3,)),str:np.ndarray (shape:(3,))})
        :return: force and torque at endpoint as named tuples in a dict in the base frame of the robot or in the stiffness frame (wrist)

          - 'force': Cartesian force on x,y,z axes in np.ndarray format
          - 'torque': Torque around x,y,z axes in np.ndarray format
        """
        return deepcopy(self._cartesian_effort) if in_base_frame else deepcopy(self._stiffness_frame_effort)

    def exit_control_mode(self, timeout=0.2):
        """
        Clean exit from advanced control modes (joint torque or velocity).
        Resets control to joint position mode with current positions if the
        advanced control commands are not send within the specified timeout
        interval.

        :type timeout: float
        :param timeout: control timeout in seconds [default: 0.2]
        """
        self.set_command_timeout(timeout)
        self.set_joint_positions(self.joint_angles())

    def tip_states(self):
        """
        Return Cartesian endpoint state for a given tip name

        :rtype: TipState object
        :return: pose, velocity, effort, effort_in_K_frame
        """
        return deepcopy(self._tip_states)

    def joint_inertia_matrix(self):
        """

        :return: joint inertia matrix (7,7)
        :rtype: np.ndarray [7x7]
        """
        return deepcopy(self._joint_inertia)

    def zero_jacobian(self):
        """
        :return: end-effector jacobian (6,7)
        :rtype: np.ndarray [6x7]
        """
        return deepcopy(self._jacobian)

    def set_command_timeout(self, timeout):
        """
        Set the timeout in seconds for the joint controller

        :type timeout: float
        :param timeout: timeout in seconds
        """
        self._pub_joint_cmd_timeout.publish(Float64(timeout))

    def set_joint_position_speed(self, speed=0.3):
        """
        Set ratio of max joint speed to use during joint position
        moves (only for move_to_joint_positions).

        Set the proportion of maximum controllable velocity to use
        during joint position control execution. The default ratio
        is `0.3`, and can be set anywhere from [0.0-1.0] (clipped).
        Once set, a speed ratio will persist until a new execution
        speed is set.

        :type speed: float
        :param speed: ratio of maximum joint speed for execution
                      default= 0.3; range= [0.0-1.0]
        """
        if speed > 0.3:
            rospy.logwarn("ArmInterface: Setting speed above 0.3 could be risky!! Be extremely careful.")
        self._speed_ratio = speed

    def set_joint_positions(self, positions):
        """
        Commands the joints of this limb to the specified positions.

        :type positions: dict({str:float}
        :param positions: dict of {'joint_name':joint_position,}
        """
        self._command_msg.names = self._joint_names
        self._command_msg.position = [positions[j] for j in self._joint_names]
        self._command_msg.mode = JointCommand.POSITION_MODE
        self._command_msg.header.stamp = rospy.Time.now()
        self._joint_command_publisher.publish(self._command_msg)

    def set_joint_velocities(self, velocities):
        """
        Commands the joints of this limb to the specified velocities.

        :type velocities: dict({str:float})
        :param velocities: dict of {'joint_name':joint_velocity,}
        """
        self._command_msg.names = self._joint_names
        self._command_msg.velocity = [velocities[j] for j in self._joint_names]
        self._command_msg.mode = JointCommand.VELOCITY_MODE
        self._command_msg.header.stamp = rospy.Time.now()
        self._joint_command_publisher.publish(self._command_msg)

    def set_joint_torques(self, torques):
        """
        Commands the joints of this limb with the specified torques.

        :type torques: dict({str:float})
        :param torques: dict of {'joint_name':joint_torque,}
        """
        self._command_msg.names = self._joint_names
        self._command_msg.effort = [torques[j] for j in self._joint_names]
        self._command_msg.mode = JointCommand.TORQUE_MODE
        self._command_msg.header.stamp = rospy.Time.now()
        self._joint_command_publisher.publish(self._command_msg)

    def set_joint_positions_velocities(self, positions, velocities):
        """
        Commands the joints of this limb using specified positions and velocities using impedance control.
        Command at time t is computed as:

        :math:`u_t= coriolis\_factor * coriolis\_t + K\_p * (positions - curr\_positions) +  K\_d * (velocities - curr\_velocities)`


        :type positions: [float]
        :param positions: desired joint positions as an ordered list corresponding to joints given by self.joint_names()
        :type velocities: [float]
        :param velocities: desired joint velocities as an ordered list corresponding to joints given by self.joint_names()
        """
        self._command_msg.names = self._joint_names
        self._command_msg.position = positions
        self._command_msg.velocity = velocities
        self._command_msg.mode = JointCommand.IMPEDANCE_MODE
        self._command_msg.header.stamp = rospy.Time.now()
        self._joint_command_publisher.publish(self._command_msg)

    def has_collided(self):
        """
        Returns true if either joint collision or cartesian collision is detected.
        Collision thresholds can be set using instance of :py:class:`franka_tools.CollisionBehaviourInterface`.
        """
        return False  # any(self._joint_collision) or any(self._cartesian_collision)

    def switchToController(self, controller_name):
        active_controllers = self._ctrl_manager.list_active_controllers(only_motion_controllers=True)
        for ctrlr in active_controllers:
            self._ctrl_manager.stop_controller(ctrlr.name)
            rospy.loginfo("ArmInterface: Stopping %s for trajectory controlling" % ctrlr.name)
            rospy.sleep(0.5)

        if not self._ctrl_manager.is_loaded(controller_name):
            self._ctrl_manager.load_controller(controller_name)
        self._ctrl_manager.start_controller(controller_name)

    def move_to_neutral(self, timeout=15.0, speed=0.15):
        """
        Command the Limb joints to a predefined set of "neutral" joint angles.
        From rosparam /franka_control/neutral_pose.

        :type timeout: float
        :param timeout: seconds to wait for move to finish [15]
        :type speed: float
        :param speed: ratio of maximum joint speed for execution
         default= 0.15; range= [0.0-1.0]
        """
        self.set_joint_position_speed(speed)
        self.move_to_joint_positions(self._neutral_pose_joints, timeout)

    def genf(self, joint, angle):
        def joint_diff():
            return abs(angle - self._joint_angle[joint])

        return joint_diff

    def move_to_joint_positions(
        self, positions, timeout=2.0, threshold=0.00085, test=None, min_traj_dur=0.1, delay=0.00
    ):
        """
        (Blocking) Commands the limb to the provided positions.
        Waits until the reported joint state matches that specified.

        This function uses a low-pass filter using JointTrajectoryService
        to smooth the movement or optionally uses MoveIt! to plan and
        execute a trajectory.

        :type positions: dict({str:float})
        :param positions: joint_name:angle command
        :type timeout: float
        :param timeout: seconds to wait for move to finish [15]
        :type threshold: float
        :param threshold: position threshold in radians across each joint when
         move is considered successful [0.00085]
        :param test: optional function returning True if motion must be aborted
        """

        if type(positions) is not dict:
            positions = {j: p for p, j in zip(positions, self._joint_names)}
        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_trajectory_controller:
            self.switchToController(self._ctrl_manager.joint_trajectory_controller)

        traj_client = JointTrajectoryActionClient(joint_names=self.joint_names())
        traj_client.clear()

        dur = []
        for j in range(len(self._joint_names)):
            dur.append(
                max(
                    abs(positions[self._joint_names[j]] - self._joint_angle[self._joint_names[j]])
                    / self._joint_limits.velocity[j],
                    min_traj_dur,
                )
            )
        duration = max(dur) / self._speed_ratio
        print("[move_to_joint_positions]: duration:", duration)
        traj_client.add_point(positions=[positions[n] for n in self._joint_names], time=duration)

        diffs = [self.genf(j, a) for j, a in positions.items() if j in self._joint_angle]

        traj_client.start()  # send the trajectory action request
        fail_msg = "ArmInterface: {0} limb failed to reach commanded joint positions.".format(self.name.capitalize())

        def test_collision():
            if self.has_collided():
                rospy.logerr(" ".join(["Collision detected.", fail_msg]))
                return True
            return False

        franka_dataflow.wait_for(
            test=lambda: test_collision()
            or (callable(test) and test() == True)
            or (all(diff() < threshold for diff in diffs)),
            timeout=max(duration, timeout),
            timeout_msg=fail_msg,
            rate=100,
            raise_on_error=False,
        )

        res = traj_client.result()
        # IPython.embed()
        if res is not None and res.error_code:
            rospy.logerr("Trajectory Server Message: {}".format(res))
            exit()

        # rospy.sleep(delay)
        # rospy.loginfo("ArmInterface: Trajectory controlling complete")

    # def execute_position_path(self, position_path, timeout=15.0,
    #                             threshold=0.00085, test=None):
    #     """
    #     (Blocking) Commands the limb to the provided positions.
    #     Waits until the reported joint state matches that specified.
    #     This function uses a low-pass filter to smooth the movement.
    #     @type positions: dict({str:float})
    #     @param positions: joint_name:angle command
    #     @type timeout: float
    #     @param timeout: seconds to wait for move to finish [15]
    #     @type threshold: float
    #     @param threshold: position threshold in radians across each joint when
    #     move is considered successful [0.008726646]
    #     @param test: optional function returning True if motion must be aborted
    #     """

    #     current_q = self.joint_angles()
    #     diff_from_start = sum([abs(a-current_q[j]) for j, a in position_path[0].items()])
    #     if diff_from_start > 0.1:
    #         raise IOError("Robot not at start of trajectory")

    #     if self._ctrl_manager.current_controller != self._ctrl_manager.joint_trajectory_controller:
    #         self.switchToController(self._ctrl_manager.joint_trajectory_controller)

    #     min_traj_dur = 0.01
    #     traj_client = JointTrajectoryActionClient(joint_names = self.joint_names())
    #     traj_client.clear()

    #     time_so_far = 0
    #     # Start at the second waypoint because robot is already at first waypoint
    #     for i in range(1, len(position_path)):
    #         q = position_path[i]
    #         dur = []
    #         for j in range(len(self._joint_names)):
    #             dur.append(max(abs(q[self._joint_names[j]] - self._joint_angle[self._joint_names[j]]) / self._joint_limits.velocity[j], min_traj_dur))

    #         time_so_far += max(dur)/self._speed_ratio
    #         traj_client.add_point(positions = [q[n] for n in self._joint_names], time = time_so_far, velocities=[0.001 for n in self._joint_names])

    #     diffs = [self.genf(j, a) for j, a in (position_path[-1]).items() if j in self._joint_angle] # Measures diff to last waypoint

    #     fail_msg = "ArmInterface: {0} limb failed to reach commanded joint positions.".format(
    #                                                   self.name.capitalize())
    #     def test_collision():
    #         if self.has_collided():
    #             rospy.logerr(' '.join(["Collision detected.", fail_msg]))
    #             return True
    #         return False

    #     # IPython.embed()
    #     traj_client.start() # send the trajectory action request

    #     franka_dataflow.wait_for(
    #         test=lambda: test_collision() or \
    #                      (callable(test) and test() == True) or \
    #                      (all(diff() < threshold for diff in diffs)),
    #         #timeout=timeout,
    #         timeout=max(time_so_far, timeout), #XXX
    #         timeout_msg=fail_msg,
    #         rate=100,
    #         raise_on_error=False
    #         )

    #     rospy.sleep(0.5)
    #     rospy.loginfo("ArmInterface: Trajectory controlling complete")

    def execute_position_path(
        self, position_path, timeout=5.0, threshold=0.00085, test=None, state_callback=False, min_traj_dur=1.0
    ):
        """
        (Blocking) Commands the limb to the provided positions.
        Waits until the reported joint state matches that specified.
        This function uses a low-pass filter to smooth the movement.

        @type positions: dict({str:float})
        @param positions: joint_name:angle command
        @type timeout: float
        @param timeout: seconds to wait for move to finish [15]
        @type threshold: float
        @param threshold: position threshold in radians across each joint when move is considered successful
        @param test: optional function returning True if motion must be aborted
        @param min_traj_dur: Minimum duration between two waypoints. NOTE: Setting this too low can result in
        `ZeroDivisionError`
        @type min_traj_dur: float
        """

        if len(position_path) > 0 and type(position_path[0]) is not dict:
            position_paths = [{j: p for j, p in zip(self._joint_names, pos.positions)} for pos in position_paths]

        current_q = self.joint_angles()
        diff_from_start = sum([abs(a - current_q[j]) for j, a in position_path[0].items()])
        print("[ExecutePositionPath] Diff from start:", diff_from_start)
        # print("[ExecutePositionPath] Current:", current_q)
        # print("[ExecutePositionPath] Start:", position_path[0])
        if diff_from_start > 0.1:
            raise IOError("[ExecutePositionPath] Robot not at start of trajectory")

        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_trajectory_controller:
            rospy.loginfo("Switching to joint trajectory controller")
            self.switchToController(self._ctrl_manager.joint_trajectory_controller)
            rospy.loginfo("Switched to joint trajectory controller")

        traj_client = JointTrajectoryActionClient(joint_names=self.joint_names())

        time_so_far = 0
        total_times = [0]
        interval_lengths = [0]

        # Start at the second waypoint because robot is already at first waypoint
        print("[ExecutePositionPath] Trajectory length:", len(position_path))
        print("[ExecutePositionPath] Speed ratio:", self._speed_ratio)
        for i in range(1, len(position_path)):
            q = position_path[i]
            dur = []
            for j in range(len(self._joint_names)):
                dur.append(
                    max(
                        abs(q[self._joint_names[j]] - self._joint_angle[self._joint_names[j]])
                        / self._joint_limits.velocity[j],
                        min_traj_dur,
                    )
                )

            interval = max(dur) / self._speed_ratio
            # print(max(dur), interval)
            interval_lengths.append(interval)

            time_so_far += interval
            total_times.append(time_so_far)

        traj_velocities = []
        # print('[ExecutePositionPath] Interval Lengths:', interval_lengths)

        for i in range(1, len(position_path)):
            q_t = position_path[i]
            positions = [q_t[n] for n in self._joint_names]

            if i < 100:
                velocities = [0.0001 * i for n in self._joint_names]

            elif i > len(position_path) - 10:
                velocities = [0.005 for n in self._joint_names]

            elif i < len(position_path) - 1:
                q_tm1 = position_path[i - 1]
                q_tp1 = position_path[i + 1]
                dt = interval_lengths[i] + interval_lengths[i + 1]
                velocities = [(q_tp1[n] - q_tm1[n]) / dt for n in self._joint_names]
                print(i, velocities)
            else:
                velocities = [0.005 for n in self._joint_names]
                print(i, velocities)
            traj_velocities.append(velocities)

            # velocities
            traj_client.add_point(positions=positions, time=total_times[i], velocities=velocities)

        diffs = [
            self.genf(j, a) for j, a in (position_path[-1]).items() if j in self._joint_angle
        ]  # Measures diff to last waypoint

        fail_msg = "ArmInterface: {0} limb failed to reach commanded joint positions.".format(self.name.capitalize())

        def test_collision():
            if self.has_collided():
                rospy.logerr(" ".join(["Collision detected.", fail_msg]))
                return True
            return False

        traj_client.start()  # send the trajectory action request
        print("execute_position_path duration:", time_so_far)

        results_callback = []
        if state_callback:
            results_callback = franka_dataflow.wait_for_with_state_callback(
                test=lambda: test_collision()
                or (callable(test) and test() == True)
                or (all(diff() < threshold for diff in diffs)),
                timeout=max(time_so_far, timeout),
                timeout_msg=fail_msg,
                rate=100,
                raise_on_error=False,
                body=self.get_state_info,
            )
            print("number log data:", len(results_callback))

        else:
            franka_dataflow.wait_for(
                test=lambda: test_collision()
                or (callable(test) and test() == True)
                or (all(diff() < threshold for diff in diffs)),
                timeout=max(time_so_far, timeout),
                timeout_msg=fail_msg,
                rate=100,
                raise_on_error=False,
            )

        # print('Arm Diff:', [diff() for diff in diffs])
        rospy.sleep(0.1)
        rospy.loginfo("ArmInterface: Trajectory controlling complete")
        return results_callback

    def execute_toppra_path(
        self,
        joint_pos_traj: PathParameterizedTrajectory,
        joint_vel_traj: PathParameterizedTrajectory,
        sample_times: Optional[Iterable[float]] = None,
        dt: float = 0.05,
        timeout: float = 5.0,
        threshold: float = 0.00085,
        test: bool = None,
    ) -> None:
        """
        (Blocking) Executed a Drake Toppra joint space trajectory.

        :param joint_pos_traj: Continous joint position trajectory.
        :param joint_vel_traj: Continous joint velocity trajectory.
        :param sample_times: The optional times to sample the trajectory at. If not given, sample the trajectory at
            equidistant points determined by `dt`.
        :param dt: The time between trajectory samples. Only used if `sample_times` is None.
        :param timeout: Seconds to wait for move to finish.
        :param threshold: Position threshold in radians across each joint when move is considered successful.
        :param test: Optional function returning True if motion must be aborted.
        """
        assert (
            joint_pos_traj.end_time() == joint_vel_traj.end_time()
        ), "Position and velocity trajectories should end at the same time."

        if sample_times is None:
            end_time = joint_pos_traj.end_time()
            num_samples = int(end_time / dt)
            sample_times = np.linspace(0.0, end_time, num_samples)
        elif abs(joint_pos_traj.end_time() - sample_times[-1]) > 0.1:
            rospy.logwarn(
                f"Last sample time ({sample_times[-1]}s) differs from trajectory end time ({joint_pos_traj.end_time()}s)"
            )

        current_q = self.joint_angles()
        diff_from_start = sum(
            [abs(pos - current_q[name]) for name, pos in zip(self._joint_names, joint_pos_traj.value(sample_times[0]))]
        )
        rospy.loginfo(f"Joint position diff from start: {diff_from_start}")
        if diff_from_start > 0.1:
            raise IOError("[ExecuteToppraPath] Robot not at start of trajectory")

        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_trajectory_controller:
            rospy.loginfo("Switching to joint trajectory controller")
            self.switchToController(self._ctrl_manager.joint_trajectory_controller)
            rospy.loginfo("Switched to joint trajectory controller")

        traj_client = JointTrajectoryActionClient(joint_names=self.joint_names())

        position_path = np.array([joint_pos_traj.value(t) for t in sample_times])
        velocity_path = np.array([joint_vel_traj.value(t) for t in sample_times])

        for positions, velocities, time in zip(position_path, velocity_path, sample_times):
            traj_client.add_point(positions=positions, time=time, velocities=velocities)

        traj_client.start()

        # Measures diff to last waypoint
        diffs = [self.genf(name, pos) for name, pos in zip(self._joint_names, position_path[-1])]

        fail_msg = "ArmInterface: {0} limb failed to reach commanded joint positions.".format(self.name.capitalize())

        def test_collision():
            if self.has_collided():
                rospy.logerr(" ".join(["Collision detected.", fail_msg]))
                return True
            return False

        franka_dataflow.wait_for(
            test=lambda: test_collision()
            or (callable(test) and test() == True)
            or (all(diff() < threshold for diff in diffs)),
            timeout=max(sample_times[-1], timeout),
            timeout_msg=fail_msg,
            rate=100,
            raise_on_error=False,
        )

        rospy.sleep(0.1)
        rospy.loginfo("ArmInterface: Trajectory controlling complete")

    def execute_moveit_trajectory(
        self,
        traj: RobotTrajectory,
        timeout: float = 5.0,
        threshold: float = 0.00085,
        test: bool = None,
    ) -> None:
        """
        (Blocking) Executed a MoveIt joint space trajectory.

        :param traj: The MoveIt trajectory to send to the robot.
        :param timeout: Seconds to wait for move to finish.
        :param threshold: Position threshold in radians across each joint when move is considered successful.
        :param test: Optional function returning True if motion must be aborted.
        """
        assert np.all(
            [joint_a == joint_b for joint_a, joint_b in zip(self._joint_names, traj.joint_trajectory.joint_names)]
        ), "Trajectory joints don't match robot joints"

        joint_traj_points = traj.joint_trajectory.points
        position_path = [p.positions for p in joint_traj_points]
        velocity_path = [p.velocities for p in joint_traj_points]
        # NOTE: Can't yet send accelerations
        # acceleration_path = [p.accelerations for p in joint_traj_points]
        times_from_start = [p.time_from_start.to_sec() for p in joint_traj_points]

        current_q = self.joint_angles()
        diff_from_start = sum([abs(pos - current_q[name]) for name, pos in zip(self._joint_names, position_path[0])])
        rospy.loginfo(f"Joint position diff from start: {diff_from_start}")
        if diff_from_start > 0.1:
            raise IOError("[ExecuteMoveItTrajectory] Robot not at start of trajectory")

        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_trajectory_controller:
            rospy.loginfo("Switching to joint trajectory controller")
            self.switchToController(self._ctrl_manager.joint_trajectory_controller)
            rospy.loginfo("Switched to joint trajectory controller")

        traj_client = JointTrajectoryActionClient(joint_names=self.joint_names())

        if len(velocity_path) > 0:
            for positions, velocities, time in zip(position_path, velocity_path, times_from_start):
                traj_client.add_point(positions=positions, time=time, velocities=velocities)
        else:
            rospy.logwarn("Velocity path is empty")
            for positions, time in zip(position_path, times_from_start):
                traj_client.add_point(positions=positions, time=time)

        traj_client.start()

        # Measures diff to last waypoint
        diffs = [self.genf(name, pos) for name, pos in zip(self._joint_names, position_path[-1])]

        fail_msg = "ArmInterface: {0} limb failed to reach commanded joint positions.".format(self.name.capitalize())

        def test_collision():
            if self.has_collided():
                rospy.logerr(" ".join(["Collision detected.", fail_msg]))
                return True
            return False

        franka_dataflow.wait_for(
            test=lambda: test_collision()
            or (callable(test) and test() == True)
            or (all(diff() < threshold for diff in diffs)),
            timeout=max(times_from_start[-1], timeout),
            timeout_msg=fail_msg,
            rate=100,
            raise_on_error=False,
        )

        rospy.sleep(0.1)
        rospy.loginfo("ArmInterface: Trajectory controlling complete")

    def get_state_info(self):
        joint_name = self._joint_names
        extra_info = {}
        extra_info["joint_position_commanded"] = self.joint_angles
        extra_info["joint_position_measured"] = convert_dict_to_joint(self.joint_angles(), joint_name)
        extra_info["joint_velocity_estimated"] = convert_dict_to_joint(self.joint_velocities(), joint_name)
        extra_info["joint_torque_measured"] = convert_dict_to_joint(self.joint_efforts(), joint_name)
        extra_info["cartesian_measured"] = convert_dict_to_wrench(self._cartesian_effort)
        extra_info["external_torque"] = self.get_external_torque()
        return extra_info

    def move_to_touch(self, positions, timeout=3.0, threshold=0.00085):
        """
        (Blocking) Commands the limb to the provided positions.

        Waits until the reported joint state matches that specified.

        This function uses a low-pass filter to smooth the movement.

        @type positions: dict({str:float})
        @param positions: joint_name:angle command
        @type timeout: float
        @param timeout: seconds to wait for move to finish [15]
        @type threshold: float
        @param threshold: position threshold in radians across each joint when
        move is considered successful [0.008726646]
        @param test: optional function returning True if motion must be aborted
        """
        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_trajectory_controller:
            self.switchToController(self._ctrl_manager.joint_trajectory_controller)

        min_traj_dur = 0.5
        traj_client = JointTrajectoryActionClient(joint_names=self.joint_names())
        traj_client.clear()

        speed_ratio = 0.1  # Move slower when approaching contact
        dur = []
        for j in range(len(self._joint_names)):
            dur.append(
                max(
                    abs(positions[self._joint_names[j]] - self._joint_angle[self._joint_names[j]])
                    / self._joint_limits.velocity[j],
                    min_traj_dur,
                )
            )
        duration = max(dur) / speed_ratio
        print("move_to_touch duration:", duration)
        traj_client.add_point(
            positions=[positions[n] for n in self._joint_names], time=duration
        )  # , velocities=[0.002 for n in self._joint_names])

        diffs = [self.genf(j, a) for j, a in positions.items() if j in self._joint_angle]
        fail_msg = "ArmInterface: {0} limb failed to reach commanded joint positions.".format(self.name.capitalize())

        traj_client.start()  # send the trajectory action request

        franka_dataflow.wait_for(
            test=lambda: self.has_collided() or (all(diff() < threshold for diff in diffs)),
            timeout=max(duration, timeout),
            timeout_msg="Move to touch complete.",
            rate=100,
            raise_on_error=False,
        )

        rospy.sleep(0.5)

        if not self.has_collided():
            rospy.logerr("Move To Touch did not end in making contact")
        else:
            rospy.loginfo("Collision detected!")

        # The collision, though desirable, triggers a cartesian reflex error. We need to reset that error
        if self._robot_mode == 4:
            self.resetErrors()

        rospy.loginfo("ArmInterface: Trajectory controlling complete")

    def resetErrors(self):
        rospy.sleep(0.5)
        pub = rospy.Publisher(
            "/franka_ros_interface/franka_control/error_recovery/goal",
            franka_msgs.msg.ErrorRecoveryActionGoal,
            queue_size=10,
        )
        rospy.sleep(0.5)
        pub.publish(franka_msgs.msg.ErrorRecoveryActionGoal())
        rospy.loginfo("Collision Reflex was reset")

    def move_from_touch(self, positions, timeout=1.5, threshold=0.00085):
        """
        (Blocking) Commands the limb to the provided positions.

        Waits until the reported joint state matches that specified.

        This function uses a low-pass filter to smooth the movement.

        @type positions: dict({str:float})
        @param positions: joint_name:angle command
        @type timeout: float
        @param timeout: seconds to wait for move to finish [15]
        @type threshold: float
        @param threshold: position threshold in radians across each joint when
        move is considered successful [0.008726646]
        @param test: optional function returning True if motion must be aborted
        """
        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_trajectory_controller:
            self.switchToController(self._ctrl_manager.joint_trajectory_controller)

        print("[move_from_touch] Desired end config")
        # print(positions)
        speed_ratio = 0.3
        min_traj_dur = 0.5
        traj_client = JointTrajectoryActionClient(joint_names=self.joint_names())
        traj_client.clear()

        dur = []
        for j in range(len(self._joint_names)):
            dur.append(
                max(
                    abs(positions[self._joint_names[j]] - self._joint_angle[self._joint_names[j]])
                    / self._joint_limits.velocity[j],
                    min_traj_dur,
                )
            )

        duration = max(dur) / speed_ratio
        print("[move_from_touch] duration:", duration)
        traj_client.add_point(positions=[positions[n] for n in self._joint_names], time=duration)

        diffs = [self.genf(j, a) for j, a in positions.items() if j in self._joint_angle]
        fail_msg = "ArmInterface: {0} limb failed to reach commanded joint positions.".format(self.name.capitalize())

        traj_client.start()  # send the trajectory action request

        franka_dataflow.wait_for(
            test=lambda: (all(diff() < threshold for diff in diffs)),
            timeout=max(duration, timeout),
            timeout_msg="Unable to complete plan!",
            rate=100,
            raise_on_error=False,
        )
        # print('[move_from_touch] Actual end config')
        # print(self.joint_angles())
        # print('[move_from_touch] Arm Diff:', [diff() for diff in diffs])
        rospy.sleep(0.5)
        rospy.loginfo("ArmInterface: Trajectory controlling complete")

    def set_cart_impedance_pose(self, pose, stiffness=None):
        if self._ctrl_manager.current_controller != self._ctrl_manager.cartesian_impedance_controller:
            self.switchToController(self._ctrl_manager.cartesian_impedance_controller)

        if stiffness is not None:
            stiffness_gains = CartImpedanceStiffness()
            stiffness_gains.x = stiffness[0]
            stiffness_gains.y = stiffness[1]
            stiffness_gains.z = stiffness[2]
            stiffness_gains.xrot = stiffness[3]
            stiffness_gains.yrot = stiffness[4]
            stiffness_gains.zrot = stiffness[5]
            self._cartesian_stiffness_publisher.publish(stiffness_gains)

        marker_pose = PoseStamped()
        marker_pose.pose.position.x = pose["position"][0]
        marker_pose.pose.position.y = pose["position"][1]
        marker_pose.pose.position.z = pose["position"][2]
        marker_pose.pose.orientation.x = pose["orientation"].x
        marker_pose.pose.orientation.y = pose["orientation"].y
        marker_pose.pose.orientation.z = pose["orientation"].z
        marker_pose.pose.orientation.w = pose["orientation"].w
        self._cartesian_impedance_pose_publisher.publish(marker_pose)

        # Do not return until motion complete
        rospy.sleep(0.1)
        while sum(map(abs, self.convertToList(self.joint_velocities()))) > 1e-2:
            rospy.sleep(0.1)

    def set_joint_impedance_config(self, q, stiffness=None):
        # Need q converted to list
        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_impedance_controller:
            self.switchToController(self._ctrl_manager.joint_impedance_controller)

        if stiffness is not None:
            stiffness_gains = JointImpedanceStiffness()
            stiffness_gains = stiffness
            self._joint_stiffness_publisher.publish(stiffness_gains)

        marker_pose = JICmd()
        marker_pose.position = q
        marker_pose.velocity = [0.005] * 7
        self._joint_impedance_publisher.publish(marker_pose)

        # Do not return until motion complete
        rospy.sleep(0.1)
        while sum(map(abs, self.convertToList(self.joint_velocities()))) > 1e-2:
            rospy.sleep(0.1)

    def set_torque(self, tau):
        raise NotImplementedError("Still working on the bugs in this!")

        switch_ctrl = True if self._ctrl_manager.current_controller != self._ctrl_manager.ntorque_controller else False
        if switch_ctrl:
            self.switchToController(self._ctrl_manager.ntorque_controller)

        torque = TorqueCmd()
        torque.torque = tau
        self._torque_controller_publisher.publish(torque)

    def execute_cart_impedance_traj(self, poses, stiffness=None):
        if self._ctrl_manager.current_controller != self._ctrl_manager.cartesian_impedance_controller:
            self.switchToController(self._ctrl_manager.cartesian_impedance_controller)

        for i in range(len(poses)):
            self.set_cart_impedance_pose(poses[i], stiffness)
            if i == 0:
                self.resetErrors()

    def execute_joint_impedance_traj(self, qs, stiffness=None):
        if self._ctrl_manager.current_controller != self._ctrl_manager.joint_impedance_controller:
            self.switchToController(self._ctrl_manager.joint_impedance_controller)

        for i in range(len(qs)):
            self.set_joint_impedance_config(qs[i], stiffness)
            if i == 0:
                self.resetErrors()

    def exert_force(self, target_wrench):
        if self._ctrl_manager.current_controller != self._ctrl_manager.force_controller:
            self.switchToController(self._ctrl_manager.force_controller)

        wrench = Wrench()
        wrench.force.x = target_wrench[0]
        wrench.force.y = target_wrench[1]
        wrench.force.z = target_wrench[2]
        wrench.torque.x = target_wrench[3]
        wrench.torque.y = target_wrench[4]
        wrench.torque.z = target_wrench[5]
        self._force_controller_publisher.publish(wrench)

    def pause_controllers_and_do(self, func, *args, **kwargs):
        """
        Temporarily stops all active controllers and calls the provided function
        before restarting the previously active controllers.

        :param func: the function to be called
        :type func: callable
        """
        assert callable(
            func
        ), "ArmInterface: Invalid argument provided to ArmInterface->pause_controllers_and_do. Argument 1 should be callable."
        active_controllers = self._ctrl_manager.list_active_controllers(only_motion_controllers=True)

        rospy.loginfo("ArmInterface: Stopping motion controllers temporarily...")
        for ctrlr in active_controllers:
            self._ctrl_manager.stop_controller(ctrlr.name)
        rospy.sleep(1.0)

        retval = func(*args, **kwargs)

        rospy.sleep(1.0)
        rospy.loginfo("ArmInterface: Restarting previously active motion controllers.")
        for ctrlr in active_controllers:
            self._ctrl_manager.start_controller(ctrlr.name)
        rospy.sleep(1.0)
        rospy.loginfo("ArmInterface: Controllers restarted.")

        return retval

    def reset_EE_frame(self):
        """
        Reset EE frame to default. (defined by
        FrankaFramesInterface.DEFAULT_TRANSFORMATIONS.EE_FRAME
        global variable defined in :py:class:`franka_tools.FrankaFramesInterface`
        source code)

        :rtype: [bool, str]
        :return: [success status of service request, error msg if any]
        """
        if self._frames_interface:

            if self._frames_interface.EE_frame_is_reset():
                rospy.loginfo("ArmInterface: EE Frame already reset")
                return

            return self.pause_controllers_and_do(self._frames_interface.reset_EE_frame)

        else:
            rospy.logwarn("ArmInterface: Frames changing not available in simulated environment")
            return False

    def set_EE_frame(self, frame):
        """
        Set new EE frame based on the transformation given by 'frame', which is the
        transformation matrix defining the new desired EE frame with respect to the flange frame.
        Motion controllers are stopped for switching

        :type frame: [float (16,)] / np.ndarray (4x4)
        :param frame: transformation matrix of new EE frame wrt flange frame (column major)
        :rtype: [bool, str]
        :return: [success status of service request, error msg if any]
        """
        if self._frames_interface:

            if self._frames_interface.frames_are_same(self._frames_interface.get_EE_frame(as_mat=True), frame):
                rospy.loginfo("ArmInterface: EE Frame already at the target frame.")
                return True

            return self.pause_controllers_and_do(self._frames_interface.set_EE_frame, frame)

        else:
            rospy.logwarn("ArmInterface: Frames changing not available in simulated environment")

    def set_EE_frame_to_link(self, frame_name, timeout=5.0):
        """
        Set new EE frame to the same frame as the link frame given by 'frame_name'
        Motion controllers are stopped for switching

        :type frame_name: str
        :param frame_name: desired tf frame name in the tf tree
        :rtype: [bool, str]
        :return: [success status of service request, error msg if any]
        """
        if self._frames_interface:
            retval = True
            if not self._frames_interface.EE_frame_already_set(self._frames_interface.get_link_tf(frame_name)):

                return self.pause_controllers_and_do(
                    self._frames_interface.set_EE_frame_to_link,
                    frame_name=frame_name,
                    timeout=timeout,
                )

        else:
            rospy.logwarn("ArmInterface: Frames changing not available in simulated environment")

    def set_collision_threshold(self, cartesian_forces=None, joint_torques=None):
        """
        Set Force Torque thresholds for deciding robot has collided.

        :return: True if service call successful, False otherwise
        :rtype: bool
        :param cartesian_forces: Cartesian force threshold for collision detection [x,y,z,R,P,Y] (robot motion stops if violated)
        :type cartesian_forces: [float] size 6
        :param joint_torques: Joint torque threshold for collision (robot motion stops if violated)
        :type joint_torques: [float] size 7
        """
        if self._collision_behaviour_interface:
            return self._collision_behaviour_interface.set_collision_threshold(
                joint_torques=joint_torques, cartesian_forces=cartesian_forces
            )
        else:
            rospy.logwarn("No CollisionBehaviourInterface object found!")

    def get_controller_manager(self):
        """
        :return: the FrankaControllerManagerInterface instance associated with the robot.
        :rtype: franka_tools.FrankaControllerManagerInterface
        """
        return self._ctrl_manager

    def get_frames_interface(self):

        """
        :return: the FrankaFramesInterface instance associated with the robot.
        :rtype: franka_tools.FrankaFramesInterface
        """
        return self._frames_interface


if __name__ == "__main__":
    rospy.init_node("test")
    r = Arm()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
