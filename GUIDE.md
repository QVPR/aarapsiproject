## Setup 
- VSCode ROS Setup: https://www.youtube.com/watch?v=RXyFSnjMd7M
- VSCode GitHub Integration: https://code.visualstudio.com/docs/sourcecontrol/github

## Information
- Git Submodules: https://gist.github.com/gitaarik/8735255, https://git-scm.com/book/en/v2/Git-Tools-Submodules

## Useful commands
- Reset Odometry: rosservice call /set_pose <press tab to populate an empty PoseStamped message, then hit enter>
- Kill ROS network (do whilst connected to jackal): killall -9 rosmaster
- Record rosbag without high-data topics: rosbag record -a -O <filename> -x "/.*raw_image.*|/.*stitched_image.*|.*image_tiles.*|.*image\d"
  - Note: for best performance and data capture, run this command onboard (otherwise some images will be lost over the network)
  - This has been scripted (in ~/Documents/bags, ./record_rosbag_tool.sh <filename>)
