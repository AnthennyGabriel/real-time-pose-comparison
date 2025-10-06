# Real-time Pose Comparison 实时姿态对比系统

本项目基于 OpenCV 和 MediaPipe 实现，支持**实时摄像头与参考视频的动作对比**，并实时给出动作相似度评分。适用于健身、舞蹈、康复等场景的动作模仿与评估。

## 功能简介

- 左侧显示实时摄像头画面，右侧顺序循环播放参考视频。
- 底部居中实时显示动作匹配度评分（每0.5秒刷新一次）。
- 新增实时反馈评价模块：
  - 当匹配度低于0.3时，界面英文提示如何调整动作以更接近参考姿态。
  - 当匹配度高于0.75时，界面英文鼓励“Similarity is very high!”。
- 关闭界面时自动导出评分数据和分析结果。
- 支持自定义参考视频、摄像头索引、模型复杂度等参数。

## 安装依赖

建议使用 Python 3.8+，并提前创建虚拟环境。

```bash
pip install -r requirements.txt
```

主要依赖：
- opencv-python
- mediapipe
- numpy
- matplotlib

## 使用方法

1. **准备参考视频**  
   将参考视频（如 `1.mp4`）放入 `videos/` 目录下。

2. **运行系统**  
   在项目根目录下执行：

   ```bash
   python src/real_time_comparator.py videos/1.mp4
   ```

   可选参数说明：
   - `--camera` 摄像头索引（默认0）
   - `--model-complexity` MediaPipe模型复杂度（0/1/2，默认1）
   - `--output-dir` 结果输出目录（默认output）

   例如：

   ```bash
   python src/real_time_comparator.py videos/1.mp4 --camera 1 --output-dir results
   ```

3. **操作说明**
   - 按 `q` 或关闭窗口退出系统。
   - 按 `s` 截图保存当前对比画面。
   - 按 `p` 暂停/继续。
   - 系统会根据动作匹配度实时给出英文反馈：
     - 匹配度低于0.3时，提示如何调整动作。
     - 匹配度高于0.75时，鼓励“Similarity is very high!”。

4. **结果导出**
   - 关闭窗口后，评分数据和分析结果会自动保存在 `output/` 目录下，包括评分曲线和详细JSON数据。

## 系统调试与常见问题

- **参考视频黑屏/卡顿？**  
  系统已优化为顺序流畅播放参考视频。若仍有问题，请检查视频格式和解码支持。

- **推送代码到GitHub失败？**  
  先执行 `git pull origin main` 合并远程内容，解决冲突后再 `git push origin main`。

- **依赖安装失败？**  
  检查 Python 版本，建议使用虚拟环境，必要时升级 pip。

## 目录结构

```
real_time_pose_comparison/
├── src/
│   └── real_time_comparator.py
├── videos/
│   └── 1.mp4
├── output/
│   └── ...（分析结果）
├── requirements.txt
└── README.md
```

## 贡献与反馈

欢迎提交 issue 或 PR 进行功能完善与问题反馈。

---

**作者：AnthennyGabriel**  
**项目地址：https://github.com/AnthennyGabriel/real-time-pose-comparison**