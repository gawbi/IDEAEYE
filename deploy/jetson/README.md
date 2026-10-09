# Jetson Orin Nano 배포 준비

현장 장비(Jetson Orin Nano, USB 카메라 2대)에서 실행하기 위한 스크립트 모음. 노트북에서 만든 모델을 ONNX 로 내보낸 뒤 Jetson 에서 TensorRT 엔진으로 변환해 실행한다.

## 순서

1. 노트북: `python deploy/jetson/export_onnx.py --cls latest:cls_mono_crackbin_effb0 --det C:/aihub_egg/runs/det_yolo11n_MONO/weights/best.pt --out deploy/jetson/models`
   - 분류기는 **고정 배치(기본 6 = 카메라 2대 × 3자리)** 로 내보내 TensorRT 에서 동적 형태 문제를 피한다.
   - 자사 영상으로 재학습한 모델이 생기면 `--cls` 만 바꿔 다시 실행.
2. Jetson: `bash deploy/jetson/setup.sh` (JetPack 6.x 기준 PyTorch·onnxruntime·pycuda 설치 확인)
3. Jetson: `bash deploy/jetson/build_trt.sh deploy/jetson/models` → `*.engine` (FP16, 필요 시 INT8 보정)
4. Jetson: `python deploy/jetson/bench_jetson.py --engine deploy/jetson/models/classifier_fp16.engine --n 200` → 계란당 지연 보고
5. Jetson: 실제 파이프라인은 `src/infer/pipeline.py` + `src/infer/adapters.py` 의 `TrtClassifier`(엔진 로드) 로 실행 — 카메라 입력은 `cv2.VideoCapture(index)` (UVC).

## 예상 성능 (실측 전 추정)

노트북 RTX 4060 에서 계란당 1.9 ms(분류 PyTorch FP16, 판 단위 GPU 크롭). Orin Nano 는 FP16 TensorRT 기준 대략 5~10배 느릴 것으로 보아 계란당 10~20 ms → 사이클당 6개 × 3시점 = 18 크롭 ≈ 0.2~0.4 초. 라인 속도가 정해지면 여유를 계산한다. 검출기는 자리가 고정이라 ROI 크롭으로 대체 가능(검출 단계 생략 시 지연 절반 이하).

## 주의

- Orin Nano 는 메모리 8 GB(공유). 모델 2개 동시 로드 + 카메라 2대 4K 디코딩이면 여유가 적다 → 카메라 해상도는 ROI 가 충분히 크면 1080p 로 낮추는 것을 검토.
- MJPEG 4K60 USB 디코딩은 CPU 부담이 큼 → GStreamer 하드웨어 디코더(`nvjpegdec`) 사용.
- 전원 모드(15W) 와 쿨링에 따라 지연이 달라지므로 벤치마크 시 `nvpmodel -q`, `jetson_clocks` 상태를 함께 기록.
