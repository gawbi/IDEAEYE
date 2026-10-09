#!/usr/bin/env bash
# Jetson Orin Nano (JetPack 6.x, Ubuntu 22.04) 환경 점검·설치. 실행: bash deploy/jetson/setup.sh
set -e
echo "== JetPack / CUDA / TensorRT 버전"
cat /etc/nv_tegra_release 2>/dev/null || true
nvcc --version 2>/dev/null | tail -1 || echo "nvcc 없음 (CUDA 경로 확인: /usr/local/cuda/bin)"
dpkg -l | grep -i tensorrt | head -3 || true
python3 - <<'EOF'
import importlib
for m in ("tensorrt", "pycuda", "onnxruntime", "torch", "cv2", "numpy"):
    try:
        mod = importlib.import_module(m); print(f"{m:12s} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"{m:12s} 없음 ({e.__class__.__name__})")
EOF
echo "== 전원 모드 / 클럭"
sudo nvpmodel -q 2>/dev/null || true
echo "== 설치 (없는 것만)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-opencv libopenblas-dev
python3 -m pip install --upgrade pip
python3 -m pip install numpy pyyaml pandas pillow pycuda onnx 2>/dev/null || true
# PyTorch for Jetson: https://developer.nvidia.com/embedded/downloads (JetPack 버전에 맞는 wheel), onnxruntime-gpu for Jetson 도 동일 페이지
echo "PyTorch/onnxruntime-gpu 는 JetPack 버전별 wheel 을 NVIDIA 다운로드 센터에서 설치하세요."
echo "== 카메라"
ls /dev/video* 2>/dev/null || echo "카메라 없음"
v4l2-ctl --list-devices 2>/dev/null || true
