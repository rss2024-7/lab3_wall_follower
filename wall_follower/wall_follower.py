#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        # self.SIDE = 1
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        # self.VELOCITY = 4.0
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        
        self.WALL_TOPIC = "/wall"
        self.WALL_OTHER_TOPIC = "/wall_other"

        self.dist_error_integral = 0
        self.angle_error_integral = 0

        self.dist_error_prev = 0
        self.angle_error_prev = 0

        self.future_dist = 0

	# Initialize your publishers and subscribers here
        self.publisher_ = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.subscription = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.lidar_callback,
            10)
        
        self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 1)
        self.line_pub_other = self.create_publisher(Marker, self.WALL_OTHER_TOPIC, 1)

        self.drive_forward()

    # Write your callback functions here 
    def drive_forward(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.VELOCITY
        drive_msg.drive.acceleration = 0.0
        drive_msg.drive.jerk = 0.0

        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.steering_angle_velocity = 0.0
        self.publisher_.publish(drive_msg)

    def lidar_callback(self, scan):
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        lookahead_dist = 7 # meters
        ranges = np.array(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        self.future_dist = scan.scan_time * self.VELOCITY

        num_points = len(ranges)

        angles = np.array([angle_min + angle_increment * i for i in range(num_points)])

        forward_dist = ranges[num_points // 2]

        if self.SIDE == 1: # Left Wall, Positive Angles
            angles_other = angles[:num_points // 2]
            ranges_other = ranges[:num_points // 2]
            angles = angles[num_points // 2:]
            ranges = ranges[num_points // 2:]
        if self.SIDE == -1: # Right Wall, Negative Angles
            angles_other = angles[num_points // 2:]
            ranges_other = ranges[num_points // 2:]
            angles = angles[:num_points // 2]
            ranges = ranges[:num_points // 2]

        # filter out less important data
        forwards_condition = np.logical_not(np.logical_and(ranges > lookahead_dist / 4, np.abs(angles) >= np.pi / 2))
        condition = np.logical_and(ranges <= lookahead_dist, forwards_condition)
        in_range_indices = np.where(condition)
        angles = angles[in_range_indices]
        ranges = ranges[in_range_indices]

        if len(angles) == 0 or len(ranges) == 0:
            self.drive_forward()
            return

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)


        # use closer points
        distances = x + y**2
        dist_thres = np.quantile(distances, 0.75)

        closer_indices = np.where(distances <= dist_thres)

        if len(y[closer_indices]) < 5:
            self.drive_forward()
            return

        x = x[closer_indices]
        y = y[closer_indices]

        if len(x) == 0 or len(y) == 0:
            self.drive_forward()
            return

        # regression to estimate wall cartesian
        # from numpy.linalg.lstsq numpy page 
        A = np.vstack([x, np.ones(len(x))]).T 
        slope, intercept = np.linalg.lstsq(A, y)[0]
        

        # IF WALL ON OPPOSITE SIDE. JANKY CODE
        if np.sign(slope) == self.SIDE and abs(slope) >= 2 and - intercept / slope * self.SIDE > 0:
            slope = - 1.0 / slope
            intercept = - intercept

        y_est = slope * x + intercept

        # regression to estimate wall polar
        # from numpy.linalg.lstsq numpy page 
        A = np.vstack([angles, np.ones(len(angles))]).T 
        m, b = np.linalg.lstsq(A, ranges)[0]

        ranges_est = m * angles + b
        
        VisualizationTools.plot_line(x, y_est, self.line_pub)

        self.drive(x, y_est, slope, intercept)


    def drive(self, x, y, slope, intercept):
        max_steer = 0.34 # radians

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.VELOCITY
        drive_msg.drive.acceleration = 0.0
        drive_msg.drive.jerk = 0.0

        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.steering_angle_velocity = 0.0

        # want to minimize cartesian slope to straighten car
        angle_error = np.arctan(slope) 
        angle_error_deriv = angle_error - self.angle_error_prev

        forward_indices = np.where(x >= self.future_dist)

        # want to minimize abs(y) to get close to wall
        dist_actual = np.mean(np.abs(y[forward_indices])) if forward_indices else 0
        dist_desired = self.DESIRED_DISTANCE
        dist_error = (dist_desired - dist_actual) * - self.SIDE
        dist_error_deriv = dist_error - self.dist_error_prev

        dist_control = dist_error + dist_error_deriv / 4 

        angle_control = angle_error

        steer_control = dist_control # + angle_control / 10

        if np.abs(dist_error) <= 0.1: steer_control = 0.0


        # TURNING
        # if a large space is empty where the estimated wall is
        within_range = np.logical_and(x >= self.future_dist, x <=  2 * self.future_dist + self.DESIRED_DISTANCE )
        if not np.any(within_range):
            steer_control = self.SIDE * np.arctan(0.3 / self.DESIRED_DISTANCE)

        # if angle wrt wall is too large, use the control the angle instead
        if abs(angle_error) >= max_steer:
            steer_control = angle_control

        # make sure the steering angle is within the allowable range
        if steer_control < - max_steer: steer_control = - max_steer
        if steer_control > max_steer: steer_control = max_steer


        # validate steering angle or else simulator will crash!
        if np.isnan(steer_control): steer_control = 0.0
        drive_msg.drive.steering_angle = steer_control

        self.publisher_.publish(drive_msg)

        self.angle_error_prev = angle_error
        self.dist_error_prev = dist_error


def main():

    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
