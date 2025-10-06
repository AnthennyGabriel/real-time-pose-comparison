实时姿态对比系统开发与调试文档

任务综述

项目概述

本项目旨在开发一个实时人体姿态对比系统，通过摄像头捕捉用户实时动作，并与预录制的参考视频进行姿态相似度分析。系统基于MediaPipe姿态检测技术，能够实时计算并显示用户动作与参考动作的相似度评分。

核心功能

1. 实时摄像头姿态捕捉：通过摄像头实时检测用户人体姿态
2. 参考视频处理：分析预录制视频中的人体姿态序列
3. 姿态相似度计算：基于关键点位置和关节角度计算实时姿态与参考姿态的相似度
4. 实时可视化：并排显示摄像头画面和最佳匹配的参考姿态
5. 结果记录与分析：保存相似度数据和生成可视化报告

技术栈

• 计算机视觉：OpenCV, MediaPipe

• 数据处理：NumPy, SciPy

• 可视化：Matplotlib, OpenCV绘图功能

• 多线程处理：Python threading, queue

基于当前代码的调试细节

1. 环境配置与依赖检查

调试任务：确保所有依赖正确安装

# 检查核心依赖
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
python -c "import mediapipe as mp; print('MediaPipe导入成功')"
python -c "import numpy as np; print(f'NumPy版本: {np.__version__}')"
python -c "import matplotlib.pyplot as plt; print('Matplotlib导入成功')"

# 检查摄像头访问权限
python -c "
import cv2
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'摄像头 {i} 可访问')
        cap.release()
    else:
        print(f'摄像头 {i} 不可访问')
"


预期结果

• 所有库应正确导入无错误

• 至少一个摄像头设备应可访问

2. 参考视频处理模块调试

调试任务：验证参考视频加载和处理功能

# 测试代码片段
def test_video_processing():
    comparator = RealTimePoseComparator("videos/test_video.mp4")
    
    # 测试视频加载
    success = comparator.load_reference_video()
    assert success, "参考视频加载失败"
    
    # 验证姿态数据提取
    assert len(comparator.reference_poses) > 0, "未提取到有效姿态数据"
    
    # 检查姿态数据结构
    sample_pose = comparator.reference_poses[0]
    assert 'keypoints' in sample_pose, "姿态数据缺少关键点信息"
    assert 'angles' in sample_pose, "姿态数据缺少角度信息"
    
    print("参考视频处理测试通过")


调试要点

• 视频文件路径是否正确

• 视频格式是否支持

• 姿态检测是否成功（检查visibility阈值）

• 关节角度计算是否正确

3. 摄像头捕获模块调试

调试任务：验证实时摄像头功能

def test_camera_capture():
    comparator = RealTimePoseComparator("videos/test_video.mp4")
    
    # 测试摄像头初始化
    comparator.camera_index = 0  # 默认摄像头
    cap = cv2.VideoCapture(comparator.camera_index)
    assert cap.isOpened(), "摄像头初始化失败"
    cap.release()
    
    # 测试单帧捕获和处理
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    processed_frame, results = comparator.process_frame(test_frame)
    
    assert processed_frame is not None, "帧处理失败"
    print("摄像头捕获测试通过")


调试要点

• 摄像头设备索引是否正确

• 帧捕获是否稳定

• 姿态检测在实时视频中的性能

4. 姿态相似度计算调试

调试任务：验证相似度算法准确性

def test_similarity_calculation():
    comparator = RealTimePoseComparator("videos/test_video.mp4")
    comparator.load_reference_video()
    
    # 创建测试姿态数据
    test_pose = comparator.reference_poses[0]  # 使用参考视频中的一帧
    
    # 测试相同姿态的相似度（应接近1.0）
    similarity = comparator.calculate_similarity(test_pose, test_pose)
    assert 0.95 <= similarity <= 1.05, f"相同姿态相似度异常: {similarity}"
    
    # 测试空姿态处理
    similarity = comparator.calculate_similarity(None, test_pose)
    assert similarity == 0.0, "空姿态处理异常"
    
    print("相似度计算测试通过")


调试要点

• 相同姿态的相似度应接近1.0

• 完全不同姿态的相似度应接近0.0

• 边界情况处理（空值、部分关键点缺失）

5. 多线程处理调试

调试任务：验证多线程数据流

def test_threading_operation():
    comparator = RealTimePoseComparator("videos/test_video.mp4")
    comparator.running = True
    
    # 启动摄像头线程
    comparator.start_camera_capture()
    
    # 等待片刻让线程运行
    time.sleep(2)
    
    # 检查数据队列
    queue_size = comparator.camera_queue.qsize()
    print(f"摄像头队列大小: {queue_size}")
    
    # 停止线程
    comparator.running = False
    if comparator.camera_processing_thread:
        comparator.camera_processing_thread.join(timeout=2.0)
    
    assert queue_size > 0, "摄像头线程未产生数据"
    print("多线程测试通过")


调试要点

• 线程启动和停止是否正常

• 数据队列是否正常工作

• 线程同步和资源释放

6. 可视化界面调试

调试任务：验证显示功能

def test_visualization():
    comparator = RealTimePoseComparator("videos/test_video.mp4")
    comparator.load_reference_video()
    
    # 测试显示画面生成
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_pose = comparator.reference_poses[0]
    
    display_frame = comparator.create_comparison_display(
        test_frame, test_pose, test_pose, 0.85, 0
    )
    
    assert display_frame.shape == (720, 1600, 3), "显示画面尺寸错误"
    
    # 测试姿态绘制
    drawn_frame = comparator.draw_pose_on_frame(test_frame, test_pose, "Test")
    assert drawn_frame is not None, "姿态绘制失败"
    
    print("可视化测试通过")


调试要点

• 显示画面布局是否正确

• 姿态绘制是否准确

• 文本和信息显示是否清晰

7. 性能优化调试

调试任务：评估和优化系统性能

def test_performance():
    comparator = RealTimePoseComparator("videos/test_video.mp4")
    
    # 性能测试参数
    test_duration = 10  # 秒
    start_time = time.time()
    frame_count = 0
    
    print("开始性能测试...")
    
    while time.time() - start_time < test_duration:
        # 模拟处理帧
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        comparator.process_frame(test_frame)
        frame_count += 1
    
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    print(f"性能测试结果: {fps:.2f} FPS")
    assert fps > 15, f"帧率过低: {fps}"
    
    print("性能测试通过")


调试要点

• 帧率是否满足实时要求（>15FPS）

• 内存使用是否合理

• 多线程是否提高性能

8. 集成测试与系统验证

调试任务：完整系统功能验证

def test_integration():
    print("开始集成测试...")
    
    # 初始化系统
    comparator = RealTimePoseComparator("videos/test_video.mp4")
    
    # 加载参考视频
    assert comparator.load_reference_video(), "参考视频加载失败"
    
    # 测试完整运行流程
    try:
        # 模拟运行5秒
        comparator.running = True
        comparator.start_camera_capture()
        
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                frame, pose = comparator.camera_queue.get(timeout=1.0)
                if pose:
                    # 查找最佳匹配
                    best_ref, similarity, idx = comparator.find_best_reference_match(pose)
                    comparator.similarity_scores.append(similarity)
            except queue.Empty:
                continue
        
        # 检查结果
        assert len(comparator.similarity_scores) > 0, "未生成相似度数据"
        print(f"平均相似度: {np.mean(comparator.similarity_scores):.3f}")
        
    finally:
        comparator.running = False
        if comparator.camera_processing_thread:
            comparator.camera_processing_thread.join(timeout=2.0)
    
    print("集成测试通过")


调试要点

• 整个系统流程是否正常

• 数据流是否连贯

• 异常处理是否健全

常见问题与解决方案

问题1: 摄像头无法访问

症状：无法打开摄像头错误
解决方案：
# 检查摄像头权限
ls -l /dev/video*
sudo usermod -a -G video $USER
# 重新登录后生效


问题2: MediaPipe检测失败

症状：无法检测到人体姿态
解决方案：
• 调整min_detection_confidence和min_tracking_confidence参数

• 确保环境光照充足

• 检查摄像头分辨率设置

问题3: 性能过低

症状：帧率低于15FPS
解决方案：
• 降低处理分辨率

• 减少MediaPipe模型复杂度

• 优化图像处理流水线

问题4: 相似度计算不准确

症状：相似度分数不符合预期
解决方案：
• 检查关键点归一化处理

• 验证角度计算算法

• 调整权重参数（位置vs角度）

验证与测试计划

单元测试

1. 视频处理模块测试
2. 姿态检测模块测试  
3. 相似度计算模块测试
4. 可视化模块测试

集成测试

1. 完整系统流程测试
2. 多线程同步测试
3. 性能压力测试

用户验收测试

1. 实际舞蹈动作对比测试
2. 不同光照条件测试
3. 多人同时检测测试

通过以上详细的调试计划和测试方案，可以确保实时姿态对比系统的稳定性和准确性，为最终用户提供可靠的服务。