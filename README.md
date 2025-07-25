# MasterPi 单目视觉距离估计与目标检测系统

本项目基于YOLO模型，支持PyTorch（ultralytics）和ncnn两种推理后端，实现了单目摄像头下的目标检测与距离估计。适用于树莓派和PC，支持摄像头实时推理、模型校准、效率对比等功能。

---

## 目录结构

- `代码/`：主程序、校准、测试等Python脚本
- `模型/best_ncnn_model/`：ncnn模型及推理脚本
- `模型/` 其他目录：训练、转换等相关文件

---

## 依赖环境

### 通用依赖
- Python >= 3.8
- numpy
- opencv-python
- matplotlib
- pillow

### PyTorch/ultralytics 相关
- torch
- ultralytics

### ncnn 相关
- ncnn Python binding（需自行编译，详见[官方文档](https://github.com/Tencent/ncnn/wiki/Python%E7%BB%91%E5%AE%9A)）

### 推荐安装命令

```bash
# 系统依赖
sudo apt update
sudo apt install python3-opencv python3-pip python3-venv python3-dev cmake

# Python依赖（建议虚拟环境）
pip install numpy matplotlib pillow torch ultralytics
# ncnn Python包需源码编译安装
```

---

## 模型准备

- PyTorch/ultralytics: 直接使用 `yolov8n.pt` 或你自己训练的 `best.pt`
- ncnn: 使用 `model.ncnn.param` 和 `model.ncnn.bin`（可用ultralytics导出或ncnn官方工具转换）

---

## 校准与测试流程

1. **采集校准数据**
   ```bash
   python3 distance_test.py --known-width 物体宽度
   # 按提示操作，按s保存测量，输入实际距离
   ```
2. **主程序推理**
```bash
   python3 distance_estimation.py --known-width 物体宽度 --model yolov8n.pt --target-class 0
```
3. **ncnn推理效率与摄像头实时推理**
```bash
   cd 模型/best_ncnn_model
   python3 ncnn_vs_pt_benchmark.py
   # 或直接摄像头实时推理
   python3 ncnn_vs_pt_benchmark.py
   ```

---

## 摄像头实时推理（ncnn）

- 运行 `ncnn_vs_pt_benchmark.py`，支持摄像头实时画面、推理耗时统计、检测框可视化。
- 按 `q` 退出。

---

## 常见问题

- **ncnn Python包无法导入**：需源码编译，见[官方文档](https://github.com/Tencent/ncnn/wiki/Python%E7%BB%91%E5%AE%9A)。
- **模型文件找不到**：请确认 `model.ncnn.param`、`model.ncnn.bin`、`yolov8n.pt` 路径正确。
- **摄像头无法打开**：检查硬件连接、驱动和权限。
- **中文字体警告**：matplotlib画图时可忽略。

---

## 参考
- [ncnn官方文档](https://github.com/Tencent/ncnn)
- [ultralytics YOLO文档](https://docs.ultralytics.com/)

如有问题欢迎提issue或联系作者。
