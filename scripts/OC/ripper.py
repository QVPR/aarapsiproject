#!/usr/bin/env python3.8

import rospy
import rosbag
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from tqdm.auto import tqdm
from tf.transformations import euler_from_quaternion

def rip_bag(bag_path, sample_rate, topics_in, printer=print, use_tqdm=True):
    '''
    Open a ROS bag and store messages from particular topics, sampling at a specified rate.
    If no messages are received, list is populated with NoneType (empties are also NoneType)
    Data is appended by row containing the raw ROS messages
    First column corresponds to sample_rate * len(data)
    Returns data (type List)

    Inputs:
    - bag_path:     String for full file path for bag, i.e. /home/user/bag_file.bag
    - sample_rate:  Float for rate at which rosbag should be sampled
    - topics_in:    List of strings for topics to extract, order specifies order in returned data (time column added to the front)
    - printer:      Method wrapper for printing (default: print)
    - use_tqdm:     Bool to enable/disable use of tqdm (default: True)
    Returns:
    List
    '''
    topics         = [None] + topics_in
    data           = []
    logged_t       = -1
    num_topics     = len(topics)
    num_rows       = 0

    # Read rosbag
    printer("Ripping through rosbag, processing topics: %s" % str(topics[1:]))

    row = [None] * num_topics
    with rosbag.Bag(bag_path, 'r') as ros_bag:
        if use_tqdm: 
            iter_obj = tqdm(ros_bag.read_messages(topics=topics))
        else: 
            iter_obj = ros_bag.read_messages(topics=topics)
        for topic, msg, timestamp in iter_obj:
            row[topics.index(topic)] = msg
            if logged_t == -1:
                logged_t    = timestamp.to_sec()
            elif timestamp.to_sec() - logged_t > 1/sample_rate:
                row[0]      = sample_rate * num_rows
                data.append(row)
                row         = [None] * num_topics
                num_rows    = num_rows + 1
                logged_t    = timestamp.to_sec()
                
    return data

def yaw_from_q(q):
    '''
    Convert geometry_msgs/Quaternion into a float yaw

    Inputs:
    - q: geometry_msgs/Quaternion
    Returns:
    - float, yaw equivalent
    '''
    return euler_from_quaternion([float(q.x), float(q.y), float(q.z), float(q.w)])[2]

rospy.init_node('test') # for rospy.Time

bag_path = '/home/claxton/aarapsi_offrobot_ws/src/aarapsi_robot_pack/data/rosbags/'
bag_name = 's4_ccw_1.bag'

# Extract ROS information:
ripped_bag = rip_bag(bag_path + bag_name, 10, ['/odom/true', '/velodyne_points'], use_tqdm=True)

# Generate raw array:
bag_array  = np.array(ripped_bag)
clean_list = []
failed_inds = []
print("Generating new array...")
for i in tqdm(range(bag_array.shape[0])):
    try:
        new_entry = [bag_array[i,0], 
                     bag_array[i,1].header.stamp.to_sec(), 
                     [  bag_array[i,1].pose.pose.position.x,              # position x
                        bag_array[i,1].pose.pose.position.y,              # position y
                        yaw_from_q(bag_array[i,1].pose.pose.orientation), # position yaw
                        bag_array[i,1].twist.twist.linear.x,              # velocity x
                        bag_array[i,1].twist.twist.linear.y,              # velocity x
                        bag_array[i,1].twist.twist.angular.z],            # velocity yaw
                     np.array(list(point_cloud2.read_points(bag_array[i,2], skip_nans=True, field_names=("x", "y", "z"))))]
        clean_list.append(new_entry)
    except:
        failed_inds.append(i)
        
print("FAILS:", len(failed_inds))
print(failed_inds)

clean_array = np.array(clean_list, dtype=object)
np.savez('/home/claxton/test', arr=clean_array)
print("DONE!")
