import rospy
from sensor_msgs.msg import LaserScan
import numpy as np

def lidarCallback(msg):
    rospy.loginfo("Received LiDAR scan with %d measurements", len(msg.ranges))
    print(msg)


def lidar_sub():
    rospy.init_node('lidar_sub', anonymous=True)#intialize nodd

    rospy.Subscriber("/scan", LaserScan, lidarCallback()) # name of topic, name of message type,

    rospy.spin()  # loop waiting for the callback function






if __name__ == '__main__':
    lidar_sub()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
