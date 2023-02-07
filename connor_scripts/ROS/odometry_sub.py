#!/usr/bin/env python

import rospy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import sys

class OdoSub:
    def __init__(self):
        self.glob_x = 0
        self.glob_y = 0
        self.glob_z = 0
        self.sub = rospy.Subscriber("/odometry/filtered", Odometry, self.callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist_msg = Twist()
        self.twist_msg.linear.x = 0.25
        self.twist_msg.angular.z = 0
        self.state = 0
        self.drive_distance = 3


    def callback(self, data):

        data_x = round(data.pose.pose.position.x, 3)
        data_y = round(data.pose.pose.position.y, 3)
        data_z = round(data.pose.pose.position.z, 3)
        
        if self.state == 0:
            self.glob_x = data_x
            self.glob_y = data_y
            self.glob_z = data_z
            rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
            rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
            rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
            self.state = 1

        if abs(self.glob_x - data_x) >= self.drive_distance:
            self.state = 2

        if self.state == 1:
            self.pub.publish(self.twist_msg)

        if self.state == 2:
            self.twist_msg.linear.x = 0
            self.pub.publish(self.twist_msg)
            rospy.loginfo(rospy.get_caller_id() + " x: %f", data_x)
            rospy.loginfo(rospy.get_caller_id() + " y: %f", data_y)
            rospy.loginfo(rospy.get_caller_id() + " z: %f", data_z)
            rospy.signal_shutdown("Goal point what reached")



if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    OdoSub()
    rospy.spin()
