#!/usr/bin/env python3

import numpy as np
from rospy_message_converter import message_converter
from pyaarapsi.core.ros_tools import rip_bag

bag1_path = '/home/claxton/part1.bag'
bag2_path = '/home/claxton/part2.bag'

data1 = list(np.array(rip_bag(bag1_path, 10, ['/vpr_nodes/path_follower/info']))[:,1])
data2 = list(np.array(rip_bag(bag2_path, 10, ['/vpr_nodes/path_follower/info']))[:,1])

data = data1 + data2
dict_list = []
for i in data:
    dict_list.append(message_converter.convert_ros_message_to_dictionary(i))

print(dict_list[0])