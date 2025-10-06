import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

class RealTimePoseComparator:
    """实时摄像头与视频姿态对比系统"""
    
    def __init__(self, reference_video_path, camera_index=0, model_complexity=1):
        """
        初始化实时姿态对比器
        
        Args:
            reference_video_path: 参考视频文件路径
            camera_index: 摄像头设备索引
            model_complexity: MediaPipe模型复杂度
        """
        # 初始化MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 创建姿态检测模型
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 视频和摄像头参数
        self.reference_video_path = reference_video_path
        self.camera_index = camera_index
        
        # 数据存储
        self.reference_poses = []  # 存储参考视频的姿态序列
        self.current_camera_pose = None  # 当前摄像头姿态
        self.similarity_scores = []  # 相似度分数历史
        
        # 性能监控
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # 线程控制
        self.video_processing_thread = None
        self.camera_processing_thread = None
        self.running = False
        
        # 数据队列
        self.camera_queue = queue.Queue(maxsize=10)
        self.reference_queue = queue.Queue(maxsize=10)
        
        print("实时姿态对比系统初始化完成")
    
    def load_reference_video(self):
        """加载并处理参考视频"""
        print(f"加载参考视频: {self.reference_video_path}")
        
        if not os.path.exists(self.reference_video_path):
            raise FileNotFoundError(f"参考视频不存在: {self.reference_video_path}")
        
        cap = cv2.VideoCapture(self.reference_video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开参考视频: {self.reference_video_path}")
        
        # 获取视频信息
        self.reference_fps = cap.get(cv2.CAP_PROP_FPS)
        self.reference_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.reference_duration = self.reference_total_frames / self.reference_fps
        
        print(f"参考视频信息: {self.reference_total_frames}帧, {self.reference_fps:.1f}FPS, {self.reference_duration:.1f}秒")
        
        # 处理参考视频帧
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔一定帧数处理一帧（提高性能）
            if frame_idx % 2 == 0:  # 每2帧处理1帧
                # 转换颜色空间并检测姿态
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 提取并存储姿态数据
                    pose_data = self.extract_pose_data(results.pose_landmarks, frame.shape)
                    pose_data['frame_index'] = frame_idx
                    pose_data['timestamp'] = frame_idx / self.reference_fps
                    pose_data['frame'] = frame.copy()  # 保存原始帧
                    self.reference_poses.append(pose_data)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"已处理参考视频帧: {frame_idx}/{self.reference_total_frames}")
        
        cap.release()
        
        if not self.reference_poses:
            raise ValueError("参考视频中未检测到有效的人体姿态")
        
        print(f"参考视频处理完成，有效姿态帧: {len(self.reference_poses)}")
        return True
    
    def extract_pose_data(self, landmarks, frame_shape):
        """从MediaPipe landmarks提取姿态数据"""
        h, w = frame_shape[:2]
        
        pose_data = {
            'keypoints': [],
            'angles': {},
            'bounding_box': None
        }
        
        # 提取关键点坐标
        keypoints = []
        for idx, lm in enumerate(landmarks.landmark):
            if lm.visibility > 0.5:  # 只考虑可见的关键点
                keypoints.append({
                    'id': idx,
                    'x': lm.x * w,
                    'y': lm.y * h,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
        
        pose_data['keypoints'] = keypoints
        
        # 计算关节角度（如果关键点足够）
        if len(keypoints) > 10:
            pose_data['angles'] = self.calculate_joint_angles(keypoints)
        
        return pose_data
    
    def calculate_joint_angles(self, keypoints):
        """计算主要关节角度"""
        angles = {}
        
        # 将关键点转换为字典以便查找
        points_dict = {kp['id']: (kp['x'], kp['y']) for kp in keypoints}
        
        # 计算肘部角度
        if 11 in points_dict and 13 in points_dict and 15 in points_dict:
            angles['left_elbow'] = self.calculate_angle(
                points_dict[11], points_dict[13], points_dict[15])
        
        if 12 in points_dict and 14 in points_dict and 16 in points_dict:
            angles['right_elbow'] = self.calculate_angle(
                points_dict[12], points_dict[14], points_dict[16])
        
        # 计算膝盖角度
        if 23 in points_dict and 25 in points_dict and 27 in points_dict:
            angles['left_knee'] = self.calculate_angle(
                points_dict[23], points_dict[25], points_dict[27])
        
        if 24 in points_dict and 26 in points_dict and 28 in points_dict:
            angles['right_knee'] = self.calculate_angle(
                points_dict[24], points_dict[26], points_dict[28])
        
        return angles
    
    def calculate_angle(self, a, b, c):
        """计算三个点之间的角度"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def start_camera_capture(self):
        """启动摄像头捕获线程"""
        self.camera_processing_thread = threading.Thread(target=self._camera_capture_loop)
        self.camera_processing_thread.daemon = True
        self.camera_processing_thread.start()
    
    def _camera_capture_loop(self):
        """摄像头捕获循环"""
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_index}")
            return
        
        print("摄像头捕获线程启动")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # 水平翻转以获得镜像效果
            frame = cv2.flip(frame, 1)
            
            # 检测姿态
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            camera_pose = None
            if results.pose_landmarks:
                camera_pose = self.extract_pose_data(results.pose_landmarks, frame.shape)
                camera_pose['timestamp'] = time.time()
                camera_pose['frame'] = frame.copy()
            
            # 将数据放入队列（非阻塞）
            if not self.camera_queue.full():
                self.camera_queue.put_nowait((frame, camera_pose))
        
        cap.release()
        print("摄像头捕获线程结束")
    
    def calculate_similarity(self, camera_pose, reference_pose):
        """计算摄像头姿态与参考姿态的相似度"""
        if not camera_pose or not reference_pose:
            return 0.0
        
        # 基于关键点位置的相似度
        position_similarity = self._calculate_position_similarity(
            camera_pose['keypoints'], reference_pose['keypoints'])
        
        # 基于关节角度的相似度
        angle_similarity = self._calculate_angle_similarity(
            camera_pose['angles'], reference_pose['angles'])
        
        # 综合相似度（加权平均）
        overall_similarity = 0.7 * position_similarity + 0.3 * angle_similarity
        
        return overall_similarity
    
    def _calculate_position_similarity(self, camera_keypoints, reference_keypoints):
        """基于关键点位置计算相似度"""
        if not camera_keypoints or not reference_keypoints:
            return 0.0
        
        # 创建关键点ID映射
        camera_dict = {kp['id']: kp for kp in camera_keypoints}
        reference_dict = {kp['id']: kp for kp in reference_keypoints}
        
        # 找到共同的关键点
        common_ids = set(camera_dict.keys()) & set(reference_dict.keys())
        if not common_ids:
            return 0.0
        
        # 计算位置差异
        distances = []
        for point_id in common_ids:
            cam_kp = camera_dict[point_id]
            ref_kp = reference_dict[point_id]
            
            # 计算欧氏距离（仅考虑x,y坐标）
            distance = np.sqrt((cam_kp['x'] - ref_kp['x'])**2 + 
                              (cam_kp['y'] - ref_kp['y'])**2)
            distances.append(distance)
        
        # 归一化距离（假设最大合理距离为图像宽度的1/4）
        max_distance = 320  # 1280/4
        normalized_distances = [min(d / max_distance, 1.0) for d in distances]
        
        # 相似度为1减去平均归一化距离
        avg_distance = np.mean(normalized_distances)
        similarity = max(0.0, 1.0 - avg_distance)
        
        return similarity
    
    def _calculate_angle_similarity(self, camera_angles, reference_angles):
        """基于关节角度计算相似度"""
        if not camera_angles or not reference_angles:
            return 0.0
        
        # 找到共同的关节角度
        common_joints = set(camera_angles.keys()) & set(reference_angles.keys())
        if not common_joints:
            return 0.0
        
        # 计算角度差异
        angle_diffs = []
        for joint in common_joints:
            diff = abs(camera_angles[joint] - reference_angles[joint])
            # 归一化角度差异（最大180度）
            normalized_diff = min(diff / 180.0, 1.0)
            angle_diffs.append(normalized_diff)
        
        # 相似度为1减去平均归一化角度差异
        avg_diff = np.mean(angle_diffs)
        similarity = max(0.0, 1.0 - avg_diff)
        
        return similarity
    
    def find_best_reference_match(self, camera_pose):
        """在参考视频中找到与当前摄像头姿态最匹配的帧"""
        if not camera_pose or not self.reference_poses:
            return None, 0.0
        
        best_similarity = 0.0
        best_reference_pose = None
        best_index = 0
        
        # 遍历参考姿态找到最佳匹配
        for i, ref_pose in enumerate(self.reference_poses):
            similarity = self.calculate_similarity(camera_pose, ref_pose)
            if similarity > best_similarity:
                best_similarity = similarity
                best_reference_pose = ref_pose
                best_index = i
        
        return best_reference_pose, best_similarity, best_index
    
    def run_comparison(self):
        """运行实时对比主循环（左摄像头，右顺序循环参考视频，底部实时评分）"""
        print("启动实时姿态对比...")
        print("按 'q' 退出，按 's' 保存截图，按 'p' 暂停/继续")
        
        # 加载参考视频
        if not self.load_reference_video():
            print("参考视频加载失败")
            return
        
        # 启动摄像头捕获
        self.running = True
        self.start_camera_capture()
        
        # 初始化显示窗口
        cv2.namedWindow('Real-time Pose Comparison', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Pose Comparison', 1600, 900)
        
        paused = False
        last_camera_pose = None
        reference_play_index = 0
        last_score_time = time.time()
        last_similarity = 0.0
        similarity_scores = []
        
        try:
            while self.running:
                if paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):
                        paused = False
                        print("继续运行")
                    elif key == ord('q'):
                        break
                    continue

                # 获取摄像头帧
                try:
                    camera_frame, camera_pose = self.camera_queue.get(timeout=0.1)
                    last_camera_pose = camera_pose
                except queue.Empty:
                    camera_pose = last_camera_pose
                    continue

                # 顺序播放参考视频帧
                reference_play_index = (reference_play_index + 1) % len(self.reference_poses)
                ref_pose = self.reference_poses[reference_play_index]

                # 每0.5秒计算一次相似度并保存
                now = time.time()
                if now - last_score_time >= 0.5 and camera_pose:
                    last_similarity = self.calculate_similarity(camera_pose, ref_pose)
                    similarity_scores.append(last_similarity)
                    last_score_time = now

                # 创建对比显示画面
                display_frame = self.create_side_by_side_display(
                    camera_frame, camera_pose, ref_pose, last_similarity
                )

                # 显示画面
                cv2.imshow('Real-time Pose Comparison', display_frame)

                # 控制参考视频播放速度（与原视频帧率同步）
                wait_time = int(1000 / self.reference_fps) if hasattr(self, 'reference_fps') and self.reference_fps > 0 else 33
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('Real-time Pose Comparison', cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"comparison_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"截图已保存: {filename}")
                elif key == ord('p'):
                    paused = True
                    print("暂停")
        except KeyboardInterrupt:
            print("程序被用户中断")
        finally:
            self.running = False
            if self.camera_processing_thread:
                self.camera_processing_thread.join(timeout=2.0)
            cv2.destroyAllWindows()
            # 导出评分数据
            self.similarity_scores = similarity_scores
            self.save_results()
            print("实时姿态对比结束")

    def create_side_by_side_display(self, camera_frame, camera_pose, reference_pose, similarity):
        """左摄像头，右参考视频，底部居中显示评分"""
        display_height = 720
        display_width = 1600
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # 摄像头画面
        camera_display = cv2.resize(camera_frame, (800, 600))
        if camera_pose:
            camera_display = self.draw_pose_on_frame(camera_display, camera_pose, "Camera")

        # 参考视频画面
        reference_display = self.create_reference_visualization(reference_pose, 0)

        # 拼接
        display_frame[60:660, 0:800] = camera_display
        display_frame[60:660, 800:1600] = reference_display

        # 标题
        cv2.putText(display_frame, "Camera", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(display_frame, "Reference Video", (900, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # 底部一行显示：左侧相似度，右侧反馈
        score_text = f"Similarity Score: {similarity:.3f}"
        feedback_text = None
        if similarity < 0.3:
            feedback_text = "Try to adjust your pose to match the reference more closely!"
        elif similarity > 0.75:
            feedback_text = "Similarity is very high!"
        # 相似度分数左侧
        cv2.putText(display_frame, score_text, (50, 700), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        # 反馈文字右侧
        if feedback_text:
            fb_x = 400  # 适当右移，避免与分数重叠
            cv2.putText(display_frame, feedback_text, (fb_x, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 2)
        return display_frame
    
    def create_comparison_display(self, camera_frame, camera_pose, reference_pose, similarity, ref_index):
        """创建对比显示画面"""
        # 创建一个大画布
        display_height = 720
        display_width = 1600
        
        # 创建黑色背景
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # 调整摄像头帧大小
        camera_display = cv2.resize(camera_frame, (800, 600))
        
        # 在摄像头帧上绘制姿态
        if camera_pose:
            camera_display = self.draw_pose_on_frame(camera_display, camera_pose, "Camera Pose")
        
        # 创建参考姿态显示（如果没有实时参考帧，显示提示）
        if reference_pose and len(self.reference_poses) > 0:
            # 创建一个代表参考姿态的示意图
            reference_display = self.create_reference_visualization(reference_pose, ref_index)
        else:
            reference_display = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(reference_display, "Waiting for pose detection...", 
                       (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 将两个画面并排放置
        display_frame[60:660, 0:800] = camera_display
        display_frame[60:660, 800:1600] = reference_display
        
        # 添加标题和分数
        cv2.putText(display_frame, "Camera", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(display_frame, "Best Match Reference Pose", (850, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # 显示相似度分数
        score_text = f"Similarity: {similarity:.3f}"
        score_color = (0, 255, 0) if similarity > 0.7 else (0, 165, 255) if similarity > 0.5 else (0, 0, 255)
        cv2.putText(display_frame, score_text, (650, 680), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
        
        # 显示FPS和帮助信息
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (50, 680), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit, 's' to save, 'p' to pause", (1000, 680), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def draw_pose_on_frame(self, frame, pose_data, title):
        """在帧上绘制姿态关键点和连接线"""
        frame_copy = frame.copy()
        
        # 绘制关键点
        for kp in pose_data['keypoints']:
            x, y = int(kp['x']), int(kp['y'])
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        
        # 绘制连接线（简化版，只绘制主要连接）
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 手臂
            (11, 23), (12, 24), (23, 24),  # 躯干
            (23, 25), (25, 27), (24, 26), (26, 28)  # 腿部
        ]
        
        kp_dict = {kp['id']: (int(kp['x']), int(kp['y'])) for kp in pose_data['keypoints']}
        
        for conn in connections:
            if conn[0] in kp_dict and conn[1] in kp_dict:
                pt1 = kp_dict[conn[0]]
                pt2 = kp_dict[conn[1]]
                cv2.line(frame_copy, pt1, pt2, (0, 255, 0), 2)
        
        # 添加标题
        cv2.putText(frame_copy, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame_copy
    
    def create_reference_visualization(self, reference_pose, ref_index):
        """创建参考姿态可视化"""
        # 如果有原始帧，显示原始帧并画骨架
        if 'frame' in reference_pose and reference_pose['frame'] is not None:
            vis_frame = cv2.resize(reference_pose['frame'], (800, 600))
            vis_frame = self.draw_pose_on_frame(vis_frame, reference_pose, f"Reference Pose #{ref_index}")
        else:
            # 没有原始帧则显示空白
            vis_frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(vis_frame, "No original frame", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 显示参考帧的时间信息
        if 'timestamp' in reference_pose:
            time_text = f"Time: {reference_pose['timestamp']:.2f}s"
            cv2.putText(vis_frame, time_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示关节角度
        if 'angles' in reference_pose and reference_pose['angles']:
            y_offset = 90
            for joint, angle in reference_pose['angles'].items():
                angle_text = f"{joint}: {angle:.1f}°"
                cv2.putText(vis_frame, angle_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25
        
        return vis_frame
    
    def save_results(self):
        """保存分析结果"""
        if not self.similarity_scores:
            print("没有可保存的结果数据")
            return
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join("output", f"comparison_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存相似度数据
        results_data = {
            "analysis_date": datetime.now().isoformat(),
            "reference_video": self.reference_video_path,
            "camera_index": self.camera_index,
            "similarity_scores": self.similarity_scores,
            "average_similarity": np.mean(self.similarity_scores) if self.similarity_scores else 0,
            "max_similarity": np.max(self.similarity_scores) if self.similarity_scores else 0,
            "min_similarity": np.min(self.similarity_scores) if self.similarity_scores else 0,
            "total_frames_processed": self.frame_count
        }
        
        with open(os.path.join(result_dir, "results.json"), "w") as f:
            json.dump(results_data, f, indent=4)
        
        # 保存相似度曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(self.similarity_scores)
        plt.xlabel('帧数')
        plt.ylabel('相似度')
        plt.title('实时姿态相似度变化曲线')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(result_dir, "similarity_curve.png"), dpi=300)
        plt.close()
        
        print(f"分析结果已保存到: {result_dir}")
        return result_dir

def main():
    """主函数：启动实时姿态对比系统"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实时摄像头与视频姿态对比系统')
    parser.add_argument('reference_video', help='参考视频文件路径')
    parser.add_argument('--camera', type=int, default=0, 
                       help='摄像头设备索引 (默认: 0)')
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2],
                       help='MediaPipe模型复杂度 (0-2, 默认: 1)')
    parser.add_argument('--output-dir', default='output', 
                       help='输出目录 (默认: output)')
    
    args = parser.parse_args()
    
    # 检查参考视频是否存在
    if not os.path.exists(args.reference_video):
        print(f"错误: 参考视频文件不存在: {args.reference_video}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 创建并运行实时姿态对比器
        comparator = RealTimePoseComparator(
            reference_video_path=args.reference_video,
            camera_index=args.camera,
            model_complexity=args.model_complexity
        )
        
        print("=" * 60)
        print("实时摄像头与视频姿态对比系统")
        print("=" * 60)
        print(f"参考视频: {args.reference_video}")
        print(f"摄像头设备: /dev/video{args.camera}")
        print("=" * 60)
        
        comparator.run_comparison()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())