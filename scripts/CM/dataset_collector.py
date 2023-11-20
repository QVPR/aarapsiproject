#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import sys
import os
import glob
import roslib

robot_x = 0
robot_y = 0
robot_z = 0
frame_num = 0
current_image = Image()
bridge = CvBridge()

class OdoSub:
    def __init__(self):
        self.glob_x = 0
        self.glob_y = 0
        self.glob_z = 0
        self.sub = rospy.Subscriber("/odometry/filtered", Odometry, self.callback)

    def callback(self, data):

        global robot_x
        global robot_y
        global robot_z

        robot_x = round(data.pose.pose.position.x, 3)
        robot_y = round(data.pose.pose.position.y, 3)
        robot_z = round(data.pose.pose.position.z, 3)

        self.glob_x = round(data.pose.pose.position.x, 3)
        self.glob_y = round(data.pose.pose.position.y, 3)
        self.glob_z = round(data.pose.pose.position.z, 3)

def callback(data):
    global current_image
    current_image = data
    


def listener(odo):

    global robot_x
    global robot_y
    global robot_z
    global bridge
    global frame_num
    global current_image

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/ros_indigosdk_occam/image0", Image, callback)
    rate = rospy.Rate(0.2) 
    while not rospy.is_shutdown():

        rospy.loginfo("Logging frame {}".format(str(frame_num).zfill(6)))
        image = bridge.imgmsg_to_cv2(current_image, "bgr8")
        cv.imwrite('/home/administrator/catkin_ws/src/jackal_odometry_testing/dataset_1/images/frame_id_{}.png'.format(str(frame_num).zfill(6)), image)
        np.savetxt('/home/administrator/catkin_ws/src/jackal_odometry_testing/dataset_1/odo/frame_id_{}.csv'.format(str(frame_num).zfill(6)), np.array([robot_x, robot_y, robot_z]),delimiter=',')
        frame_num += 1

        rate.sleep()

if __name__ == '__main__':
    odo = OdoSub()
    try:
        listener(odo)
    except rospy.ROSInterruptException:
        pass