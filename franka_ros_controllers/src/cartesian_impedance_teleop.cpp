// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include <thread> 
#include <mutex>
#include <stdio.h>
#include <signal.h>    

#include <Eigen/Dense>
#include <chrono>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include "examples_common.h"

#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "std_msgs/Bool.h"


/**
 * @example cartesian_impedance_control.cpp
 * An example showing a simple cartesian impedance controller without inertia shaping
 * that renders a spring damper system where the equilibrium is the initial configuration.
 * After starting the controller try to push the robot around and try different stiffness levels.
 *
 * @warning collision thresholds are set to high values. Make sure you have the user stop at hand!
 */

Eigen::Vector3d dummy_position;
Eigen::Quaterniond dummy_orientation;
std::mutex mtx, mtx_null;
bool is_enabled;
bool is_enabled_prev = false;
Eigen::Matrix<double, 6, 1> dummy_twist;
bool done = false;

void signal_callback_handler(int signum) {
    std::cout << "CTRL+C interrupted. " << std::endl;
    // Terminate program
    if (signum == SIGINT) {
        done = true;
    }
    exit(signum);
}

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{  
    mtx.lock();
    dummy_position << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    dummy_orientation.w() = msg->pose.orientation.w;
    dummy_orientation.x() = msg->pose.orientation.x;
    dummy_orientation.y() = msg->pose.orientation.y;
    dummy_orientation.z() = msg->pose.orientation.z;
    mtx.unlock();
}

void deadmanCallback(const std_msgs::Bool::ConstPtr& msg)
{
    is_enabled = msg->data;
}

void twistCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
    dummy_twist << msg->linear.x, msg->linear.y, msg->linear.z, msg->angular.x, msg->angular.y, msg->angular.z;
}

int main(int argc, char** argv) {
    // Check whether the required arguments were passed
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
        return -1;
    }
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;
    ros::Subscriber subPose = n.subscribe("/dummy_arm_pose", 1000, poseCallback);
    ros::Subscriber subTwist = n.subscribe("/dummy_arm_twist", 1000, twistCallback);
    ros::Subscriber subDeadman = n.subscribe("/enable_move", 1000, deadmanCallback);
    ros::Rate loop_rate(100);

    // Compliance parameters
    const double translational_stiffness{300.0};
    const double rotational_stiffness{10.0};
    Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
    stiffness.setZero();
    stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
    stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
    damping.setZero();
    damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) *
                                     Eigen::MatrixXd::Identity(3, 3);
    damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) *
                                         Eigen::MatrixXd::Identity(3, 3);

    // Joint null space compliance parameters
    Eigen::MatrixXd k_gains_wrist(3, 3);
    k_gains_wrist << 25.0, 0.0, 0.0,
                     0.0, 15.0, 0.0,
                     0.0, 0.0, 5.0;
    Eigen::MatrixXd d_gains_wrist(3, 3);
    d_gains_wrist << 3.0, 0.0, 0.0,
                     0.0, 2.5, 0.0,
                     0.0, 0.0, 1.5;
    Eigen::Matrix<double, 7, 7> k_gains, d_gains;
    k_gains.setZero();
    d_gains.setZero();
    k_gains.topLeftCorner(4, 4) << 20.0 * Eigen::MatrixXd::Identity(4, 4);
    d_gains.topLeftCorner(4, 4) << 3.0 * Eigen::MatrixXd::Identity(4, 4);
    k_gains.bottomRightCorner(3, 3) << 20.0 * Eigen::MatrixXd::Identity(3, 3);
    d_gains.bottomRightCorner(3, 3) << 3.0 * Eigen::MatrixXd::Identity(3, 3);
    // std::cout << k_gains_wrist << std::endl;

    try {
        // connect to robot
        franka::Robot robot(argv[1]);
        robot.automaticErrorRecovery();
        setDefaultBehavior(robot);
        // load the kinematics and dynamics model
        franka::Model model = robot.loadModel();

        franka::RobotState initial_state;

        // equilibrium point is the initial position
        Eigen::Affine3d initial_transform;
        Eigen::Vector3d position_initial;
        Eigen::Quaterniond orientation_initial;

        Eigen::Vector3d position_initial_dummy;
        Eigen::Quaterniond orientation_initial_dummy;

        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;

        Eigen::Matrix<double, 7, 7> null_space_coeffs;
        Eigen::Matrix<double, 7, 1> q0;
        Eigen::Matrix<double, 7, 7> inertia;
        Eigen::Matrix<double, 6, 7> jacobian;
        Eigen::VectorXd tau_null(7);

        // set collision behavior
        robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

        // define callback for the torque control loop
        std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
            impedance_control_callback = [&](const franka::RobotState& robot_state,
                                            franka::Duration period) -> franka::Torques 
        {
            auto start = std::chrono::high_resolution_clock::now();
            // get state variables
            std::array<double, 7> coriolis_array = model.coriolis(robot_state);
            std::array<double, 42> jacobian_array =
                model.zeroJacobian(franka::Frame::kEndEffector, robot_state);
            std::array<double, 49> inertia_array = model.mass(robot_state);

            // convert to Eigen
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
            jacobian = Eigen::Map<const Eigen::Matrix<double, 6, 7>>(jacobian_array.data());
            // Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
            Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
            inertia = Eigen::Map<const Eigen::Matrix<double, 7, 7>>(inertia_array.data());
            // Eigen::Map<const Eigen::Matrix<double, 7, 7>> inertia(inertia_array.data());

            position = transform.translation();
            orientation = transform.linear();

            Eigen::Vector3d relative_error;
            Eigen::Vector3d dummy_relative_error;
            Eigen::Affine3d dummy_transform(dummy_orientation.normalized().toRotationMatrix());

            // compute error to desired equilibrium pose
            // position error
            Eigen::Matrix<double, 6, 1> error;
            error.head(3) << -(dummy_position - position_initial_dummy) + (position - position_initial);

            // orientation error
            // "difference" quaternion
            if (orientation_initial.coeffs().dot(orientation.coeffs()) < 0.0) {
                orientation.coeffs() << -orientation.coeffs();
            }
            if (orientation_initial_dummy.coeffs().dot(dummy_orientation.coeffs()) < 0.0) {
                dummy_orientation.coeffs() << -dummy_orientation.coeffs();
            }
            // "difference" quaternion
            Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_initial);
            Eigen::Quaterniond dummy_error_quaternion(dummy_orientation.inverse() * orientation_initial_dummy);

            relative_error << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
            // Transform to base frame
            relative_error << transform.linear() * relative_error;

            dummy_relative_error << dummy_error_quaternion.x(), dummy_error_quaternion.y(), dummy_error_quaternion.z();
            dummy_relative_error << dummy_transform.linear() * dummy_relative_error;

            error.tail(3) << dummy_relative_error - relative_error;

            Eigen::Matrix<double, 6, 1> vel_error;
            if (is_enabled) {
                vel_error <<  dummy_twist - jacobian * dq;
            }
            else {
                vel_error << - jacobian * dq;
            }
            double ratio = 0.8;

            // compute control
            Eigen::VectorXd tau_task(7), tau_d(7);

            // Spring damper system with damping ratio=1
            // tau_task << jacobian.transpose() * (-stiffness * error - (1 - ratio) * damping * (jacobian * dq) + ratio * damping * vel_error);
            tau_task << jacobian.transpose() * (-damping * (jacobian * dq));
            mtx_null.lock();
            tau_null << null_space_coeffs * (-k_gains * (q - q0) - d_gains * dq);
            mtx_null.unlock();
            // std::cout << tau_null.transpose() << std::endl;
            tau_d << tau_task + coriolis;

            std::array<double, 7> tau_d_array{};
            Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
 
            // std::cout << "Time taken by function: "
                //  << duration.count() << " microseconds" << std::endl;
            return tau_d_array;
        };

        std::function<void()> null_space_control = [&]() {
            while(!done) {
                Eigen::LLT<Eigen::Matrix<double, 7, 7>> llt_inertia(inertia);
                Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt_jjt(jacobian * llt_inertia.solve(Eigen::MatrixXd::Identity(7, 7)) * jacobian.transpose());
                null_space_coeffs << Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * llt_jjt.solve(Eigen::MatrixXd::Identity(6, 6)) * jacobian * llt_inertia.solve(Eigen::MatrixXd::Identity(7, 7));
                // Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(jacobian * llt_inertia.solve(Eigen::MatrixXd::Identity(7, 7)) * jacobian.transpose());
                // std::cout << svd.singularValues().transpose() << std::endl;
            }
        };

        initial_state = robot.readOnce();
        initial_transform = Eigen::Matrix4d::Map(initial_state.O_T_EE.data());
        position_initial = initial_transform.translation();
        orientation_initial = initial_transform.linear();

        q0 = Eigen::Map<Eigen::Matrix<double, 7, 1>>(initial_state.q.data());
        inertia = Eigen::Map<Eigen::Matrix<double, 7, 7>>(model.mass(initial_state).data());
        jacobian = Eigen::Map<const Eigen::Matrix<double, 6, 7>>(model.zeroJacobian(franka::Frame::kEndEffector, initial_state).data());
        null_space_coeffs.setZero();

        position_initial_dummy = dummy_position;
        orientation_initial_dummy = dummy_orientation;

        // start real-time control loop
        std::cout << "WARNING: Collision thresholds are set to high values. "
                << "Make sure you have the user stop at hand!" << std::endl
                << "After starting try to push the robot and see how it reacts." << std::endl
                << "Press Enter to continue..." << std::endl;
        std::cin.ignore();
        // robot.control(impedance_control_callback);
        std::thread th_control([&](){
            robot.control(impedance_control_callback);
            });
        std::thread th_null([&](){null_space_control();});

        while (ros::ok())
        {
            signal(SIGINT, signal_callback_handler);
            if (!is_enabled)
            {
                position_initial_dummy = dummy_position;
                orientation_initial_dummy = dummy_orientation;
            }
            if (!is_enabled && is_enabled_prev)
            {
                position_initial = position;
                orientation_initial = orientation;
            }
            is_enabled_prev = is_enabled;
            // std::cout << is_enabled << std::endl;
            ros::spinOnce();
            loop_rate.sleep();
        }
        // ros::spin();
        th_control.join();
        th_null.join();

        } catch (const franka::Exception& ex) {
            // print exception
            std::cout << ex.what() << std::endl;
        }

    return 0;
}
