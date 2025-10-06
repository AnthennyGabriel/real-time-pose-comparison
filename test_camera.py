import cv2

def test_camera_access():
    # 测试所有可能的摄像头设备
    for i in range(0, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✓ 摄像头 /dev/video{i} 可用")
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"test_camera_{i}.jpg", frame)
                print(f"  测试图像已保存: test_camera_{i}.jpg")
            cap.release()
        else:
            print(f"✗ 摄像头 /dev/video{i} 不可用")

if __name__ == "__main__":
    test_camera_access()
