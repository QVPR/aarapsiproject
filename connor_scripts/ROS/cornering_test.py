#!/usr/bin/env python

import rospy
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import sys

class OdoSub:
    def __init__(self):
        self.goal_x = 3
        self.goal_y = -3
        self.goal_z = 0
        # self.goal_theta = math.radians(-90)
        self.state = 0
        self.sub = rospy.Subscriber("/odometry/filtered", Odometry, self.callback)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        # rospy.set_param('/move_base/base_global_planner', "SBPLPlannerROS")
        # rospy.set_param('/move_base/base_local_planner', "TebLocalPlannerROS")
        rospy.set_param('/move_base/xy_goal_tolerance', 0.02)
        rospy.set_param('/move_base/controller/p', 5)
        # rospy.set_param('/move_base/yaw_goal_tolerance', 0.02)
        # rospy.set_param('/move_base/controller_frequency', 30)
        # rospy.set_param('/move_base/planner_patience', 30)
        # rospy.set_param('/move_base/controller_patience', 40)
        # rospy.set_param('/move_base/oscillation_distance', 0.05)
        # rospy.set_param('/move_base/max_vel_x', 0.05)
        # rospy.set_param('/move_base/max_vel_y', 0.05)
        self.client.wait_for_server()
        # self.twist_msg = Twist()
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = "base_link"
        self.goal.target_pose.pose.position.x = self.goal_x
        self.goal.target_pose.pose.position.y = self.goal_y
        self.goal.target_pose.pose.orientation.w = 1.0
        # self.goal.target_pose.pose.orientation.z = math.sin(self.goal_theta/2)
        # self.goal.target_pose.pose.orientation.w = math.cos(self.goal_theta/2)
        self.client.send_goal(self.goal)
        self.client.wait_for_result()
        # self.twist_msg.linear.x = 0.25
        # self.twist_msg.angular.z = 0
        # self.drive_distance = 3


    def callback(self, data):

        data_x = round(data.pose.pose.position.x, 3)
        data_y = round(data.pose.pose.position.y, 3)
        data_z = round(data.pose.pose.position.z, 3)
        
        if self.state == 0:
            rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
            rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
            rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
            self.state = 1

        if (self.state == 1) and (abs(data_x-self.goal_x)<0.05) and (abs(data_y-self.goal_y)<0.05):
            rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
            rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
            rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
            self.state = 2


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
