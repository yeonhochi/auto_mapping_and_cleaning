#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped

import numpy as np

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # /map 구독
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # 마커 발행자
        self.marker_pub = self.create_publisher(Marker, 'marker_waypoints', 10)

        # Nav2 액션 클라이언트 생성
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 맵 데이터
        self.map_data = None
        self.map_array = None  # 2D numpy array로 변환한 결과
        self.map_received = False
        self.resolution = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0

        # 웨이포인트 관련
        self.waypoints = []           # 전체 순회할 웨이포인트
        self.visited_waypoints = []   # 방문 완료
        self.unvisited_waypoints = [] # 방문 전

        self.current_waypoint_idx = 0
        self.is_navigating = False

        # 타이머: 주기적으로 상태 확인
        self.timer = self.create_timer(1.0, self.timer_callback)

    def map_callback(self, msg: OccupancyGrid):
        """
        맵(OccupancyGrid)을 받으면 numpy array로 변환,
        지그재그 경로(벽에서 떨어진 셀만)를 웨이포인트로 생성.
        """
        if not self.map_received:
            self.get_logger().info("Received map data, converting to array...")

            # OccupancyGrid -> numpy array
            self.map_data = msg
            self.map_array = self.occupancygrid_to_array(msg)
            self.height, self.width = self.map_array.shape
            self.resolution = msg.info.resolution
            self.origin_x = msg.info.origin.position.x
            self.origin_y = msg.info.origin.position.y

            self.map_received = True

            # 지그재그로 웨이포인트 생성
            # (row=0→1→2... / 짝수행은 left->right, 홀수행은 right->left)
            self.waypoints = self.generate_zigzag_waypoints(step_col=3, step_row=3, wall_threshold=5)

            self.unvisited_waypoints = list(self.waypoints)

            # 마커 초기 표시
            self.update_markers()
            self.get_logger().info(f"Total waypoints: {len(self.waypoints)}")

    def timer_callback(self):
        """주기적으로 웨이포인트 순회(네비게이션) 로직을 수행."""
        if self.map_received and not self.is_navigating and self.waypoints:
            self.get_logger().info("Start navigating to the first waypoint...")
            self.is_navigating = True
            self.current_waypoint_idx = 0
            self.send_next_goal()

    def occupancygrid_to_array(self, occupancy_grid: OccupancyGrid):
        """
        OccupancyGrid -> 2D numpy array (shape: [height, width])
        -1(Unknown), 0(Free), 100(Obstacle) 값 유지
        """
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        data = np.array(occupancy_grid.data, dtype=np.int8)
        data_2d = np.reshape(data, (height, width))
        return data_2d

    def is_near_wall(self, col, row, threshold=8):
        """
        해당 (row, col)이 지도에서 장애물(값=100)과
        'threshold' 셀 이내에 있는지 검사.
        True면 '벽 근처'임.
        """
        # 맵 범위
        H, W = self.map_array.shape
        for dy in range(-threshold, threshold+1):
            for dx in range(-threshold, threshold+1):
                nr = row + dy
                nc = col + dx
                if 0 <= nr < H and 0 <= nc < W:
                    if self.map_array[nr, nc] == 100:
                        return True
        return False

    def generate_zigzag_waypoints(self, step_col=13, step_row=13, wall_threshold=9):
        """
        지그재그 방식으로 맵의 모든 셀을 훑으며,
        1) 값이 0(이동 가능) 이고,
        2) 벽에서 'wall_threshold' 이상 떨어진 셀
        만 웨이포인트로 삼는다.
        
        - step_col: 열(가로) 방향 샘플링 간격
        - step_row: 행(세로) 방향 샘플링 간격
        """
        waypoints = []
        H, W = self.map_array.shape

        for row in range(0, H, step_row):
            # 짝수 행이면 왼->오, 홀수 행이면 오->왼
            if (row // step_row) % 2 == 0:
                col_range = range(0, W, step_col)
            else:
                col_range = range(W-1, -1, -step_col)

            for col in col_range:
                cell_value = self.map_array[row, col]
                if cell_value == 0:
                    # 벽 근처인지 검사
                    if not self.is_near_wall(col, row, wall_threshold):
                        # 유효한 셀이면 world 좌표로 변환
                        wx, wy = self.grid_to_world(col, row)
                        waypoints.append((wx, wy))

        return waypoints

    def grid_to_world(self, col, row):
        """OccupancyGrid (col, row) -> 실제 map 좌표 (x, y) 변환."""
        # origin + (col + 0.5)*res, (row + 0.5)*res
        # row=세로방향(-> y), col=가로방향(-> x)
        wx = self.origin_x + (col + 0.5) * self.resolution
        wy = self.origin_y + (row + 0.5) * self.resolution
        return (wx, wy)

    def send_next_goal(self):
        """현재 waypoint 인덱스에 해당하는 Goal을 Nav2에 전송."""
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info("All waypoints are done!")
            self.is_navigating = False
            return

        x, y = self.waypoints[self.current_waypoint_idx]

        # PoseStamped 목표 생성
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.w = 1.0  # 회전 없이

        self.get_logger().info(f"Sending goal {self.current_waypoint_idx} -> (x={x:.2f}, y={y:.2f})")

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_pose

        self.nav_to_pose_client.wait_for_server()
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            nav_goal,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Nav2가 goal을 수락했는지 확인하는 콜백."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected!")
            self.is_navigating = False
            return

        self.get_logger().info("Goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Nav2에서 피드백이 올 때마다 불리는 콜백 (옵션)."""
        # 진행률 등을 파악 가능
        pass

    def get_result_callback(self, result_future):
        """Goal 완료 후 호출되는 콜백 (성공/실패)."""
        status = result_future.result().status
        if status == 4:  # 4 = SUCCEEDED
            self.get_logger().info(f"Goal {self.current_waypoint_idx} succeeded!")
            visited_pt = self.waypoints[self.current_waypoint_idx]
            if visited_pt in self.unvisited_waypoints:
                self.unvisited_waypoints.remove(visited_pt)
                self.visited_waypoints.append(visited_pt)

            # 마커 갱신
            self.update_markers()

            # 다음 웨이포인트
            self.current_waypoint_idx += 1
            self.send_next_goal()
        else:
            self.get_logger().warn(f"Goal failed with status={status}")
            self.is_navigating = False
          
    def update_markers(self):
        """방문/미방문 웨이포인트를 서로 다른 색의 점으로 표시."""
        # 방문한 웨이포인트 (초록색 점)
        visited_marker = Marker()
        visited_marker.header.frame_id = "map"
        visited_marker.header.stamp = self.get_clock().now().to_msg()
        visited_marker.type = Marker.POINTS
        visited_marker.action = Marker.ADD
        visited_marker.scale.x = 0.1   # 점의 크기 (가로)
        visited_marker.scale.y = 0.1   # 점의 크기 (세로)
        visited_marker.color.a = 1.0
        visited_marker.color.r = 0.0
        visited_marker.color.g = 0.8
        visited_marker.color.b = 0.0

        for (x, y) in self.visited_waypoints:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            visited_marker.points.append(p)

        # 미방문 웨이포인트 (빨간색 점)
        unvisited_marker = Marker()
        unvisited_marker.header.frame_id = "map"
        unvisited_marker.header.stamp = self.get_clock().now().to_msg()
        unvisited_marker.type = Marker.POINTS
        unvisited_marker.action = Marker.ADD
        unvisited_marker.scale.x = 0.1
        unvisited_marker.scale.y = 0.1
        unvisited_marker.color.a = 1.0
        unvisited_marker.color.r = 1.0
        unvisited_marker.color.g = 0.0
        unvisited_marker.color.b = 0.0

        for (x, y) in self.unvisited_waypoints:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            unvisited_marker.points.append(p)

        visited_marker.id = 0
        unvisited_marker.id = 1

        self.marker_pub.publish(visited_marker)
        self.marker_pub.publish(unvisited_marker)



def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
