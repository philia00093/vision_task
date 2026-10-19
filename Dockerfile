# 树莓派（arm64）上的「末端」镜像：只有 numpy + opencv + pillow，**没有 torch**。
#
# 推理在电脑上做：树莓派用 --infer-backend http --wire binary
# （上行一帧图像帧、下行 160 字节结果帧，见 protocol.py）。
#
# 构建：docker build -t vision-edge:pi .
# 运行：见 docker-compose.pi.yml（设备、网络、卷都在那里配好了）
#
# 基础镜像是 debian bookworm slim，arm64v8 官方支持；树莓派 4B/5 的 64 位系统直接能用。
FROM python:3.11-slim-bookworm

# 想在树莓派桌面上看预览窗口，就用 opencv-python（带 Qt），构建时加：
#   docker compose -f docker-compose.pi.yml build --build-arg OPENCV_PKG=opencv-python
ARG OPENCV_PKG=opencv-python-headless

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VISION_TASK_ROOT=/app

# libgl1/libglib2.0-0 是 OpenCV 的运行时依赖（headless 版其实不需要，装上省得来回折腾）
# v4l-utils 用来在容器里排查摄像头：v4l2-ctl --list-devices / --list-formats-ext
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 v4l-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 单独一层装依赖：只要不改这三行，改代码不会触发重新下载 wheel
RUN pip install --no-cache-dir numpy "$OPENCV_PKG" pillow

# 拷代码。.dockerignore 已经排除了 800MB 训练图 / 1.6GB 数据集 / runs，
# 但保留了 vision_yolo_train/images/steps（4.7MB，阶段识别的参考图要用）。
COPY . .

# 容器里的项目根就是 /app（edge/project.py 会在这里找到 vision_yolo_train/）
ENTRYPOINT ["python", "edge/run_edge.py"]
CMD ["--infer-backend", "http", "--wire", "binary", "--no-window"]
