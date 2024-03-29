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
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/vesc/input/navigation")
        self.declare_parameter("side", "-1")
        self.declare_parameter("velocity", "1.0")
        self.declare_parameter("desired_distance", "default")

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        
        self.WALL_TOPIC = "/wall"
        self.DISTANCE_TOPIC = "/dist"
        self.ANGLE_TOPIC = "/angle"

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
        self.dist_line_pub = self.create_publisher(Marker, self.DISTANCE_TOPIC, 1)
        self.angle_pub = self.create_publisher(Marker, self.ANGLE_TOPIC, 1)

        self.drive_forward()

    # Write your callback functions here 
    def drive_forward(self):

        # self.get_logger().info('Drive Forward')
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.VELOCITY
        
        drive_msg.drive.acceleration = 0.0
        drive_msg.drive.jerk = 0.0

        drive_msg.drive.steering_angle = -0.05
        drive_msg.drive.steering_angle_velocity = 0.0
        self.publisher_.publish(drive_msg)

    def lidar_callback(self, scan):
        # for testing safety controller
        # self.drive_forward()
        # return

        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        MAX_STEER = 0.34 # radians
        
        TURN_RADIUS = 0.3 / np.sin(MAX_STEER)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.VELOCITY
        drive_msg.drive.acceleration = 0.0
        drive_msg.drive.jerk = 0.0

        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.steering_angle_velocity = 0.0

        lookahead_dist = 7 # meters
        ranges = np.array(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        self.future_dist = scan.scan_time * self.VELOCITY

        num_points = len(ranges)

        angles = np.array([angle_min + angle_increment * i for i in range(num_points)])

        angle_thres = np.pi / 12

        if self.SIDE == 1: # Left Wall, Positive Angles
            ranges = ranges[angles >= angle_thres]
            angles = angles[angles >= angle_thres]
        if self.SIDE == -1: # Right Wall, Negative Angles
            ranges = ranges[angles <= - angle_thres]
            angles = angles[angles <= - angle_thres]

        # filter out less important data
        forwards_condition = np.logical_not(np.logical_and(ranges > lookahead_dist / 4, np.abs(angles) >= np.pi / 2))
        forwards_condition = np.abs(angles) <= np.pi / 2
        forwards_condition = True
        condition = np.logical_and(ranges <= lookahead_dist, forwards_condition)
        in_range_indices = np.where(condition)
        angles = angles[in_range_indices]
        ranges = ranges[in_range_indices]

        if len(angles) == 0 or len(ranges) == 0:
            self.drive_forward()
            return

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        y = y[x <= 2 * self.DESIRED_DISTANCE]
        x = x[x <= 2 * self.DESIRED_DISTANCE]


        # use closer points
        distances = x**2 + y**2
        dist_thres = np.quantile(distances, 0.5)

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
        

        # # IF WALL ON OPPOSITE SIDE. JANKY CODE to pass autograder
        # if np.sign(slope) == self.SIDE and abs(slope) >= 2 and - intercept / slope * self.SIDE > 0:
        #     slope = - 1.0 / slope
        #     intercept = - 1 / intercept

        y_est = slope * x + intercept

        # regression to estimate wall polar
        # from numpy.linalg.lstsq numpy page 
        A = np.vstack([angles, np.ones(len(angles))]).T 
        m, b = np.linalg.lstsq(A, ranges)[0]

        ranges_est = m * angles + b
        
        VisualizationTools.plot_line(x, y_est, self.line_pub, frame="laser")

        # ----------------------------------------------------
        # DRIVING
        # ----------------------------------------------------
        y = y_est

        # want to minimize cartesian slope to straighten car
        angle_error = np.arctan(slope) 
        angle_error_deriv = angle_error - self.angle_error_prev

        forward_indices = np.where(np.logical_and(x >= self.future_dist, x <= self.VELOCITY / 2))
        forward_indices = np.where(x >= self.future_dist)

        # want to minimize abs(y) to get close to wall
        dist_actual = np.mean(np.abs(y[forward_indices])) if forward_indices else self.DESIRED_DISTANCE
        dist_desired = self.DESIRED_DISTANCE
        dist_error = (dist_desired - dist_actual) * - self.SIDE
        dist_error_deriv = dist_error - self.dist_error_prev

        dist_control = dist_error + dist_error_deriv / 3

        angle_control = angle_error # self.VELOCITY # + angle_error_deriv / 4

        steer_control = dist_control # + angle_control / 10

        if np.abs(dist_error) <= 0.1: steer_control = 0.0


        # TURNING
        angles = np.array([angle_min + angle_increment * i for i in range(num_points)])
        scan_ranges = np.array(scan.ranges)

        # defining turn fov as the section of the lidar from -max turning angle to +max turning angle
        # the side (left or right) half of a turn fov
        side_block_ranges = scan_ranges[np.where(np.logical_and(self.SIDE * angles >= 0, self.SIDE * angles <= MAX_STEER))]
        # the entire turn fov
        all_block_ranges = scan_ranges[np.abs(angles) <= MAX_STEER]

        # TURN TO AVOID HITTING SOMETHING
        avoid_distance = 1.25 * max(self.DESIRED_DISTANCE, TURN_RADIUS)
        avoid_distance = 1.12 * (self.DESIRED_DISTANCE + TURN_RADIUS)
        if np.quantile(all_block_ranges, 0.75) <= avoid_distance or np.quantile(side_block_ranges, 0.75) <= avoid_distance:
            steer_control = - self.SIDE * MAX_STEER
        if (all_block_ranges[0] + all_block_ranges[-1]) / 2 < 0.5:
             steer_control = - self.SIDE * MAX_STEER

        # TURNING CORNERS
        # look 45-60 degs to side. see if have enough space
        view_angle = np.pi/4
        turn_lookahead_condition = np.where(
                np.logical_and(
                    self.SIDE * angles >= np.pi / 4 - np.pi / 30, 
                    np.abs(angles) <= np.pi / 3 + np.pi / 30
                    )
                )
        turn_lookahead_ranges = scan_ranges[turn_lookahead_condition]
        turn_clearance = np.mean(turn_lookahead_ranges)

        # make sure there is space to turn into
        y = np.abs(np.sin(angles) * scan_ranges)
        turn_space = y[np.where(np.logical_and(self.SIDE * angles >= 0, np.abs(angles) <= np.pi / 2))]

        # combine all corner turn conditions
            # if lateral_clearance >=  self.DESIRED_DISTANCE and \
        required_turn_clearance = 4 * TURN_RADIUS
        required_turn_clearance = 2.5 * self.DESIRED_DISTANCE
        if np.quantile(turn_space, 0.90) >= required_turn_clearance \
            and turn_clearance >= TURN_RADIUS :
            # steer_control = self.SIDE / np.arctan(0.3 / self.DESIRED_DISTANCE)
            steer_control = self.SIDE * np.arcsin(0.3 / self.DESIRED_DISTANCE)


        # if angle wrt wall is too large, use the control the angle instead
        if abs(angle_error) >= MAX_STEER:
            steer_control = angle_control

        # make sure the steering angle is within the allowable range
        if steer_control < - MAX_STEER: steer_control = - MAX_STEER
        if steer_control > MAX_STEER: steer_control = MAX_STEER


        # validate steering angle or else simulator will crash!
        if np.isnan(steer_control): steer_control = 0.0

        # - 0.05 to adjust for fact that car naturally turns left
        drive_msg.drive.steering_angle = steer_control - 0.05

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
    
