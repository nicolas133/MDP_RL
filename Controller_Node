import ropsy
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from pid_controller.pid import PID

class controllerNode:
    def __init__(self):
        ropsy.init('contoller_node')

        #parameters for position controll

        # Establish Publishers for sending proceseed info
        self.speed_pub= rospy.Publisher('/speed_cmd',Float64,que_size=1)#memory of one message
        self.steering_pub= rospy.Publisher('/steering_cmd',Float64,que_size=1)

        # Establish Subscribers for recieving info for control
        self.EFK_sub = ropsy.Subscriber('/odom',Odometry,self.Controller_callback)

        #Create Contollers
        self.Speed_PID = PID(kp=1,ki=1,kd=1)
        self.Steering_PID = PID(kp=1,ki=1,kd=1)
        # contol position with speed and steering

    def Controller_callback(self,msg):
        rospy.loginfo("Received IMU data with %d measurements", len(msg.ranges))
        print(msg)

        speed_error=self.Speed_error(msg)
        position_error=self.Position_error(msg)
        steering_error = self.Steering_error(msg)
        new_S= self.Speed_PID(speed_error)
        new_SS= self.Steering_PID(steering_error)
        self.speed_pub(new_S)
        self.steering_pub(new_SS)

    def Speed_error(self,msg):
        currentV=msg.twist.twist.linear.x
        V_optimal=1
        speed_error=currentV-V_optimal
        return speed_error
    def Position_error(self,msg):
        self.waypoints = []  # insert list of cordinates
        self.wayindex = 0
        self.distancethresh = 1  # edit later
        current_way = self.waypoints(self.wayindex)
        # get current position
        x_cord=msg.pose.pose.position.x
        y_cord=msg.pose.pose.position.y
        z_cord=msg.pose.pose.position.z
        position_error = math.sqrt((x_cord - current_way[0]) ** 2 + (y_cord - current_way[1]) ** 2)
        if position_error < self.distance_thresh:
            self.wayindex = (self.wayindex + 1) 
        return position_error

    def Steering_error(self, msg,x_cord,y_cord):
        current_orientation = msg.pose.pose.orientation
        current_orientation_euler = tf.transformations.euler_from_quaternion([
        current_orientation.x,
        current_orientation.y,
        current_orientation.z,
        current_orientation.w])

        next_waypoint2 = self.waypoints[self.wayindex]
        angle_waypoint = math.atan2(next_waypoint2[1] - y_cord, next_waypoint2[0] - x_cord)
        current_steering_angle= current_orientation_euler[2]
        steering_error=angle_waypoint-current_steering_angle

        # Make sure the steering error is in the range [-pi, pi]
        while steering_error > math.pi:
            steering_error -= 2 * math.pi
        while steering_error < -math.pi:
            steering_error += 2 * math.pi

        return steering_error



    def run(self):
        ropsy.spin()

if __name__ == '__main__':
     node = controllerNode()
     node.run()
