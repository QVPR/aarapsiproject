#!/usr/bin/env python

import rospy
import math
import copy
import random
# import argparse

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
import sys

class OdoSub:
    def __init__(self):
        self.glob_x = 0
        self.glob_y = 0
        self.glob_z = 0
        self.sub = rospy.Subscriber("/odometry/filtered", Odometry, self.callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.odoPub = rospy.Publisher("/odometry/uncertain", Odometry, queue_size=1)
        self.pointPub = rospy.Publisher("/place_estimate/point1", Marker, queue_size=1)
        self.twist_msg = Twist()
        self.uncertain_msg = Odometry()
        self.the_point_msg = Marker()
        self.twist_msg.linear.x = 0.5
        self.twist_msg.angular.z = 0
        self.state = 0
        self.drive_distance = 3*random.random()
        self.goal_distance = 5
        self.uncertain_goal = 5
        self.reset = 0


    def callback(self, data):

        data_x = round(data.pose.pose.position.x, 3)
        data_y = round(data.pose.pose.position.y, 3)
        data_z = round(data.pose.pose.position.z, 3)
        
        self.uncertain_msg = copy.copy(data)
        # self.uncertain_msg.pose.covariance[0] = 4
        # self.uncertain_msg.pose.covariance[7] = 1
        x_variance = (-6*math.cos((0.2)*(data_x-3))+6.2)**2
        y_variance = (-4*math.cos((0.12)*(data_x-3))+4.1)**2
        # x_variance = (-8*math.cos((0.23)*(data_x-3))+8.1)**2
        # y_variance = (-5*math.cos((0.2)*(data_x-3))+5.1)**2

        loc_x, loc_y = self.rnd_ellipse_pnts(math.sqrt(x_variance), math.sqrt(y_variance), data_x, data_y)
        self.the_point_msg.header.frame_id = "odom"
        self.the_point_msg.pose.position.x = loc_x
        self.the_point_msg.pose.position.y = loc_y
        self.the_point_msg.pose.position.z = 0
        self.the_point_msg.scale.x = 0.1
        self.the_point_msg.scale.y = 0.1
        self.the_point_msg.scale.z = 0.1
        self.the_point_msg.color.a = 1.0
        self.the_point_msg.color.r = 0.0
        self.the_point_msg.color.g = 1.0
        self.the_point_msg.color.b = 0.0
        self.the_point_msg.pose.orientation.w = 1.0
        self.the_point_msg.type = self.the_point_msg.CUBE
        self.the_point_msg.action = self.the_point_msg.ADD

        # rospy.loginfo(rospy.get_caller_id() + " x: %f", loc_x)
        # rospy.loginfo(rospy.get_caller_id() + " y: %f", loc_y)        

        self.uncertain_msg.pose.covariance = [
            x_variance, 0, 0, 0, 0, 0,
            0, y_variance, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ]
        self.odoPub.publish(self.uncertain_msg)
        self.pointPub.publish(self.the_point_msg)


        if self.state == 0:
            self.reset = 0
            self.glob_x = data_x
            self.glob_y = data_y
            self.glob_z = data_z
            # rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
            # rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
            # rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
            self.drive_distance = 3*random.random()
            self.state = 1

        if (data_x >= self.drive_distance) and self.reset == 0:
            self.state = 2

        if self.state == 1:
            self.pub.publish(self.twist_msg)

        if self.state == 2:
            self.uncertain_goal = (self.goal_distance - loc_x)+data_x
            self.state = 1
            self.reset = 1

        if data_x >= self.uncertain_goal:
            rospy.loginfo(rospy.get_caller_id() + " decision distance: %f", self.drive_distance)
            rospy.loginfo(rospy.get_caller_id() + " uncertain goal: %f", self.uncertain_goal)
            rospy.loginfo(rospy.get_caller_id() + " Error: %f", self.uncertain_goal-self.goal_distance)
            self.state = 3


        if self.state == 3:
            if data_x <= 0:
                self.state = 0
                self.twist_msg.linear.x = 0.5
                self.pub.publish(self.twist_msg)
            else:
                self.twist_msg.linear.x = -0.75
                self.pub.publish(self.twist_msg)
                # if self.reset == 0:
                    # rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
                    # rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
                    # rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
            self.reset = 1
                # rospy.signal_shutdown("Goal point what reached")

    def rnd_ellipse_pnts(self, uncertainty_x, uncertainty_y, x, y):

        rp = math.sqrt(random.random())
        phi = random.random()*2*math.pi
        x_c = rp*math.cos(phi)
        y_c = rp*math.sin(phi)
        x_e = x_c * uncertainty_x/2
        y_e = y_c * uncertainty_y/2
        a = (x_e*math.cos(0) - y_e*math.sin(0)) + x
        b = (x_e*math.sin(0) + y_e*math.cos(0)) + y

        return a, b



if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    OdoSub()
    rospy.spin()
