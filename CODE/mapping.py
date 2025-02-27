#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatusArray
import numpy as np
import random
from rclpy.qos import qos_profile_system_default
import matplotlib.pyplot as plt
import yaml

class FrontierExplorationNode(Node):
    def __init__(self):
        super().__init__('frontier_exploration_node')

        self.map_subscriber = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        self.goal_status_subscriber = self.create_subscription(
            GoalStatusArray,
            '/follow_path/_action/status',  
            self.goal_status_callback,
            10)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_system_default)
        self.goal_publisher = self.create_publisher(
            PoseStamped, 'goal_pose', 10)

        self.map_array = None
        self.map_metadata = None
        self.goal_reached = True  
        self.robot_x = 0.0
        self.robot_y = 0.0

        self.timer = self.create_timer(5.0, self.timer_callback)

    def map_callback(self, msg: OccupancyGrid):
        # OccupancyGrid -> numpy array
        data = np.array(msg.data, dtype=np.int8)
        self.map_array = data.reshape((msg.info.height, msg.info.width))
        self.map_metadata = msg.info

    def goal_status_callback(self, msg: GoalStatusArray):
        if msg.status_list:
            current_status = msg.status_list[-1].status
            if current_status == 3:
                # succeeded
                self.goal_reached = True

            elif current_status == 4:
                # aborted
                self.goal_reached = True
 
            elif current_status == 5:
                self.goal_reached = True

            elif current_status == 6:
                self.goal_reached = True

            else:
                self.goal_reached = False
                self.get_logger().info(f"목표 상태={current_status}")

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def timer_callback(self):
        if self.map_array is None or self.map_metadata is None:
            return

        # 만약 이미 goal을 향해 가는 중(goal_reached=False)이면 skip
        if not self.goal_reached:
            self.get_logger().info("목표에 도달하지 않았습니다.")
            return

        # frontier 감지
        frontiers = self.detect_frontiers()
        if not frontiers:
            self.get_logger().info("탐색 가능한 구역이 없습니다.")
            # self.save_map_to_file()  # 더 이상 갈 목표가 없으면 맵 저장
            return

        # 벽 근처 제거
        valid_frontiers = []
        for (col, row) in frontiers:
            if not self.is_near_wall(col, row, self.map_array, 4):
                valid_frontiers.append((col, row))

        if not valid_frontiers:
            self.get_logger().info("남은 후보가 없습니다.")
            # self.save_map_to_file()  # 더 이상 갈 목표가 없으면 맵 저장
            return

        # 랜덤 선택
        goal_xy = self.select_goal(valid_frontiers)
        if goal_xy is None:
            self.get_logger().info("목표 선택 실패.")
            return

        # goal publish
        self.publish_goal(goal_xy)

    def detect_frontiers(self):
        H, W = self.map_array.shape
        frontiers = []
        for row in range(1, H-1):
            for col in range(1, W-1):
                if self.map_array[row, col] == -1:
                    neighbors = self.map_array[row-1:row+2, col-1:col+2]
                    if (neighbors == 0).any():
                        frontiers.append((col, row))
        return frontiers

    def is_near_wall(self, col, row, map_data, threshold):
        H, W = map_data.shape
        for dy in range(-threshold, threshold+1):
            for dx in range(-threshold, threshold+1):
                nr = row + dy
                nc = col + dx
                if 0 <= nr < H and 0 <= nc < W:
                    if map_data[nr, nc] == 100:
                        return True
        return False

    def select_goal(self, frontiers):
        if not frontiers:
            return None
        chosen = random.choice(frontiers)
        col, row = chosen
        # 픽셀 -> meter
        res = self.map_metadata.resolution
        origin_x = self.map_metadata.origin.position.x
        origin_y = self.map_metadata.origin.position.y

        # 실제 좌표
        real_x = col*res + origin_x
        real_y = row*res + origin_y
        return (real_x, real_y)

    def publish_goal(self, goal):
        # goal=(real_x, real_y) 실제 좌표
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"

        goal_msg.pose.position.x = goal[0]
        goal_msg.pose.position.y = goal[1]
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f"Goal => x={goal[0]:.3f}, y={goal[1]:.3f}")
        self.goal_reached = False

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

