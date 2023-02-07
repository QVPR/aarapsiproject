#!/usr/bin/env python

import rospy
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import sys

class OdoSub:
    def __init__(self):
        self.goal_x = 3
        self.goal_y = 0
        self.goal_z = 0
        # self.goal_theta = math.radians(-90)
        self.state = 0
        self.sub = rospy.Subscriber("/odometry/filtered", Odometry, self.callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist_msg = Twist()
        self.twist_msg.linear.x = 0
        self.twist_msg.angular.z = 0
        # self.twist_msg.angular.w = 0
        self.start_theta = 0
        # self.drive_distance = 3


    def callback(self, data):

        data_x = round(data.pose.pose.position.x, 3)
        data_y = round(data.pose.pose.position.y, 3)
        data_z = round(data.pose.pose.position.z, 3)
        data_orientation = (data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        roll, pitch, data_theta = euler_from_quaternion(data_orientation)
        
        if self.state == 0:
            rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
            rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
            rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)

            self.goal_theta = math.atan2(self.goal_y-data_y, self.goal_x-data_x)

            self.start_theta = data_theta
            self.start_ori = data_orientation

            angle_diff = self.goal_theta - self.start_theta
            if angle_diff > math.pi:
                angle_diff -= 2*pi
                self.theta_sign = -1
            elif angle_diff < -math.pi:
                angle_diff += 2*pi
                self.theta_sign = 1
            else:
                self.theta_sign = math.copysign(1, angle_diff)

            rospy.loginfo(rospy.get_caller_id() + " goal theta: %f", self.goal_theta)
            rospy.loginfo(rospy.get_caller_id() + " start theta: %f", self.start_theta)
            rospy.loginfo(rospy.get_caller_id() + " theta sign: %f", self.theta_sign)

            # quat = quaternion_from_euler(0, 0, self.theta_sign*0.2)
            self.twist_msg.linear.x = 0
            self.twist_msg.angular.z = self.theta_sign*max(1.2*angle_diff, 0.05)



            # self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            # self.client.wait_for_server()
            # self.goal = MoveBaseGoal()
            # self.goal.target_pose.header.frame_id = "base_link"
            # self.goal.target_pose.pose.position.x = data_x
            # self.goal.target_pose.pose.position.y = data_y
            # self.goal.target_pose.pose.orientation.z = math.sin(theta/2)
            # self.goal.target_pose.pose.orientation.w = math.cos(theta/2)
            # self.client.send_goal(self.goal)
            # self.client.wait_for_result()

            self.state = 1

        angle_diff = self.goal_theta - data_theta
        if angle_diff > math.pi:
            angle_diff -= 2*pi
        elif angle_diff < -math.pi:
            angle_diff += 2*pi

        if self.state == 1 and abs(data_theta-self.goal_theta) < 0.01:
            rospy.loginfo(rospy.get_caller_id() + " Done Rotating")
            self.twist_msg.linear.x = 0.5
            self.twist_msg.angular.z = 0
            # self.twist_msg.angular.w = 0
            self.state = 2


        if self.state == 1:
            self.twist_msg.angular.z = self.theta_sign*max(1.2*angle_diff, 0.05)
            self.pub.publish(self.twist_msg)


        if (abs(data_x-self.goal_x)<0.05) and (abs(data_y-self.goal_y)<0.05):
            self.state = 3

        if self.state == 2:
            self.pub.publish(self.twist_msg)


        if self.state == 3:
            rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
            rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
            rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
            self.twist_msg.linear.x = 0
            self.pub.publish(self.twist_msg)
            rospy.signal_shutdown("Goal point what reached")


        # if abs(self.glob_x - data_x) >= self.drive_distance:
        #     self.state = 2

        # if self.state == 1:
        #     self.pub.publish(self.twist_msg)

        # if self.state == 2:
        #     self.twist_msg.linear.x = 0
        #     self.pub.publish(self.twist_msg)
        #     rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
        #     rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
        #     rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
        #     rospy.signal_shutdown("Goal point what reached")



if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    OdoSub()
    rospy.spin()
