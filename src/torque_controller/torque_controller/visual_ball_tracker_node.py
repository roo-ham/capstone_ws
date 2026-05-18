#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import cv2
import numpy as np
import threading
from multiprocessing import shared_memory

class BallTrackerNode(Node):
    def __init__(self):
        super().__init__('ball_tracker_node')
        
        # 1. USB 카메라 초기화 (640x480)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error("카메라를 열 수 없습니다. 연결을 확인하세요.")

        # 타겟 색상 정의 (RGB 공간)
        self.RED_TARGET = np.array([230, 60, 60], dtype=np.int32)
        self.YELLOW_TARGET = np.array([180, 210, 80], dtype=np.int32)
        
        # 마커의 바닥 프레임 좌표
        self.marker_floor_pts = np.array([
            [-0.13,  0.115], # 0: Top-Left
            [ 0.13,  0.115], # 1: Top-Right
            [-0.13, -0.115], # 2: Bottom-Left
            [ 0.13, -0.115]  # 3: Bottom-Right
        ], dtype=np.float32)
        
        self.homography_matrix = None

        # 2. 칼만 필터 변수 초기화
        self.kf_lock = threading.Lock()
        self.dt = 1.0 / 300.0
        self.X = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0

        # 모델
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)

        # (기존 __init__ 코드 내부에 추가 및 변경)
        self.last_ball_pos = None
        self.last_cam_time = None

        # 4차원 시스템에 맞춘 H 및 R 행렬 설정
        self.H = np.eye(4, dtype=np.float64)  # 4x4 단위행렬로 변경
        
        # 모델링 오차
        self.Q = np.array([
            [0.0001, 0,    0.0,    0.0],     # 위치 불확실성 (카메라 반영을 위해 살짝 열어둠)
            [0,    0.0001, 0.0,    0.0],
            [0.0,  0,    0.01,  0.0],     # 속도 불확실성 (공이 튕기거나 가속될 때 빠르게 반응하도록 1.0~5.0 수준 부여)
            [0,    0.0,    0,    0.01]
        ], dtype=np.float64)
        
        # 측정 오차
        self.R = np.array([
            [0.0001, 0,    0,    0],
            [0,    0.0001, 0,    0],
            [0,    0,    0.01,  0],
            [0,    0,    0,    0.01]
        ], dtype=np.float64)

        # 3. 공유 메모리 설정
        self.shm_name = 'ball_state_shm'
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=32)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
        self.shm_array = np.ndarray((4,), dtype=np.float64, buffer=self.shm.buf)
        self.shm_array[:] = 0.0

        # UI 및 디버그 창 공유용 변수
        self.current_ui_img = None
        self.current_mask_img = None  # 추가: 마스크 디버그 이미지용 변수
        self.ui_lock = threading.Lock()

        # 4. 타이머 설정
        self.cam_timer = self.create_timer(1.0 / 30.0, self.process_camera)
        self.kf_timer = self.create_timer(self.dt, self.process_kalman_filter)

        self.get_logger().info("Ball Tracker Node가 시작되었습니다.")

    def process_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # 2. H, S, V 채널 분리 (S가 채도)
        h, s, v = cv2.split(hsv)
        
        # 3-A. [방법 1] 고정된 수치만큼 채도 더하기
        # cv2.add를 사용하면 255를 넘어가도 0으로 돌아가지 않고 255(최대치)로 고정(Clip)해줍니다.
        s_new = cv2.add(s, 0)
        v_new = cv2.add(v, 0)
        
        # 3-B. [방법 2] 비율로 채도를 올리고 싶다면 아래 코드를 사용하세요 (예: 1.5배)
        # factor = 1.5
        # s_new = np.clip(s * factor, 0, 255).astype(np.uint8)
        
        # 4. 수정된 S 채널을 H, V 채널과 다시 병합
        hsv_new = cv2.merge([h, s_new, v_new])
        
        # 5. HSV를 다시 BGR(원본 색상 공간)로 변환
        result_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        # BGR -> RGB 변환 및 int32 캐스팅
        img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB).astype(np.int32)
        R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

        # 빨간색 계열 추출
        red_dist = np.abs(R - self.RED_TARGET[0]) + np.abs(G - self.RED_TARGET[1]) + np.abs(B - self.RED_TARGET[2])
        red_mask = (red_dist <= 110).astype(np.uint8) * 255
        
        # 노란색 계열 추출
        yellow_dist = np.abs(R - self.YELLOW_TARGET[0]) + np.abs(G - self.YELLOW_TARGET[1]) + np.abs(B - self.YELLOW_TARGET[2])
        yellow_mask = (yellow_dist <= 110).astype(np.uint8) * 255

        # Erode 적용
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)

        # === 픽셀 확인용 마스크 이미지 생성 ===
        # 검은 바탕에 인식된 픽셀만 색칠 (OpenCV는 BGR 포맷 사용)
        mask_display = np.zeros((480, 640, 3), dtype=np.uint8) + result_img
        mask_display[red_mask > 0] = [0, 0, 255]      # 빨강 (BGR)
        mask_display[yellow_mask > 0] = [0, 255, 255] # 노랑 (BGR)
        # ==================================

        # 무게중심 구하기
        red_cents = self.get_centroids(red_mask, max_count=1)
        yellow_cents = self.get_centroids(yellow_mask, max_count=4)

        ball_floor_pos = None

        if len(yellow_cents) == 4:
            # 마커 정렬 및 Homography
            yellow_cents.sort(key=lambda p: p[1])
            top_two = sorted(yellow_cents[:2], key=lambda p: p[0])
            bottom_two = sorted(yellow_cents[2:], key=lambda p: p[0])
            ordered_yellow_pts = np.array([top_two[0], top_two[1], bottom_two[0], bottom_two[1]], dtype=np.float32)

            self.homography_matrix, _ = cv2.findHomography(ordered_yellow_pts, self.marker_floor_pts)

        if self.homography_matrix is not None and len(red_cents) == 1:
            ball_cam_pt = np.array([[[red_cents[0][0], red_cents[0][1]]]], dtype=np.float32)
            ball_floor_pt = cv2.perspectiveTransform(ball_cam_pt, self.homography_matrix)
            ball_floor_pos = (ball_floor_pt[0][0][0], ball_floor_pt[0][0][1])

            # --- 4차원 Z 벡터 생성을 위한 속도 미분치 계산 ---
            current_time = self.get_clock().now().nanoseconds / 1e9  # 초 단위 변환
            vx_meas, vy_meas = 0.0, 0.0

            if self.last_ball_pos is not None and self.last_cam_time is not None:
                dt_cam = current_time - self.last_cam_time
                if dt_cam > 0:
                    # 현재 위치 - 이전 위치 / 시간 간격
                    vx_meas = (ball_floor_pos[0] - self.last_ball_pos[0]) / dt_cam
                    vy_meas = (ball_floor_pos[1] - self.last_ball_pos[1]) / dt_cam

            # 다음 루프를 위해 현재 값 저장
            self.last_ball_pos = ball_floor_pos
            self.last_cam_time = current_time
            # --------------------------------------------------

            # 칼만 필터 측정(Update) 단계 수행 (4차원 Z 벡터)
            with self.kf_lock:
                # Z = [x, y, vx, vy] 4차원 입력 생성
                Z = np.array([ball_floor_pos[0], ball_floor_pos[1], vx_meas, vy_meas], dtype=np.float64)

                np.copyto(self.shm_array, Z)
                
                y_res = Z - self.H @ self.X
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.X = self.X + K @ y_res
                # self.P = (np.eye(4) - K @ self.H) @ self.P
                self.P = np.eye(4)
                self.measured = True

        # UI 이미지 생성
        ui_img = np.zeros((300, 200, 3), dtype=np.uint8)
        def map_coord(val, is_y=False):
            mapped = int((val + 0.2) / 0.4 * 200)
            return 200 - mapped if is_y else mapped

        for mx, my in self.marker_floor_pts:
            cv2.circle(ui_img, (map_coord(mx), map_coord(my, True)), 5, (0, 255, 255), -1)

        if ball_floor_pos:
            bx, by = ball_floor_pos
            cv2.circle(ui_img, (map_coord(bx), map_coord(by, True)), 6, (0, 0, 255), -1)
            
            with self.kf_lock:
                bx, by = self.X[0], self.X[1]
                cv2.circle(ui_img, (map_coord(bx), map_coord(by, True)), 5, (255, 255, 255), 1)
                vx, vy = self.X[2], self.X[3]
            cv2.putText(ui_img, f"X: {bx:.3f} m", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(ui_img, f"Y: {by:.3f} m", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(ui_img, f"Vx: {vx:.3f} m/s", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(ui_img, f"Vy: {vy:.3f} m/s", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(ui_img, "Ball Not Found", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.line(ui_img, (0, 200), (200, 200), (255, 255, 255), 2)

        # 메인 스레드로 넘겨주기 위해 변수에 복사
        with self.ui_lock:
            self.current_ui_img = ui_img.copy()
            self.current_mask_img = mask_display.copy() # 마스크 디버그 화면 저장

    def process_kalman_filter(self):
        with self.kf_lock:
            Q = self.Q
            self.X = self.F @ self.X
            self.P = self.F @ self.P @ self.F.T + Q
            # np.copyto(self.shm_array, self.X)

    def get_centroids(self, mask, max_count):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_count]
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        return centroids

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.shm.close()
        self.shm.unlink()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # ROS 2 Spin을 백그라운드 스레드에서 실행
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        # 메인 스레드에서 OpenCV 창 렌더링
        while rclpy.ok():
            with node.ui_lock:
                img_to_show = node.current_ui_img.copy() if node.current_ui_img is not None else None
                mask_to_show = node.current_mask_img.copy() if node.current_mask_img is not None else None
            
            # 도식화 UI 표시
            # if img_to_show is not None:
            #     cv2.imshow('Floor Frame Tracker UI', img_to_show)
            
            # # # 색상 픽셀 확인용 창 표시 (추가됨)
            if mask_to_show is not None:
                cv2.imshow('Color Mask Debug', mask_to_show)
            
            # # ESC 키(27) 누르면 종료
            if cv2.waitKey(30) & 0xFF == 27: 
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()