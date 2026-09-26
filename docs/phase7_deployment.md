# Phase 7 배포 준비 결과 (2026-09-24, 노트북 RTX 4060 기준)

대상 장비·PLC 기종이 미정이라 **TensorRT 변환과 실제 PLC 통신은 미실시**. 이 문서는 ONNX 변환과 파이프라인 지연 측정, 안전 동작 구현 상태를 기록한다. 실제 장비에서 `scripts/export_and_benchmark.py` 를 다시 실행하면 같은 형식으로 재측정된다.

## 1. 산출물

| 항목 | 위치 |
|---|---|
| 검출기 ONNX (YOLO11n, 640) | `C:/aihub_egg/runs/det_yolo11n_MONO/weights/best.onnx` |
| 분류기 ONNX (EfficientNet-B0 256, 33클래스) + 클래스 목록 | `experiments/20260924_185030_benchmark_MONO/classifier.onnx`, `classifier_classes.json` |
| 파이프라인 | `src/infer/pipeline.py` (검출 → 크롭 → 분류 → 판정 → PLC), 어댑터 `src/infer/adapters.py` |
| PLC | 인터페이스 `src/plc/base.py`, mock `src/plc/mock.py` (Modbus TCP 매핑 초안: `docs/plc_interface.md`) |
| 판정 로그 | `outputs/infer_logs/judgments.jsonl` 형식 (판 시퀀스, 계란별 bbox·라벨·점수·배출 여부, 단계별 ms) → 재학습 데이터 원천 |

## 2. 지연 (판 이미지 100장, 판당 계란 12.1개, mock PLC 포함)

| 단계 | p50 | p95 | 비고 |
|---|---:|---:|---|
| 검출 (Ultralytics PyTorch FP16) | 12.1 ms | 14.4 ms | 순수 추론은 1.9 ms, 나머지는 전·후처리 오버헤드 |
| 분류 (ONNX Runtime CUDA, 계란별 배치 1) | 66.1 ms | 106.7 ms | 계란 12개 × 5.5 ms. **배치 추론으로 바꾸면 계란당 0.6 ms** (PyTorch 배치 16 측정치) → 판당 ~8 ms 가능 |
| PLC 신호 (mock) | 0.0 ms | 0.0 ms | 실제 Modbus 는 수 ms |
| **판 합계** | **78.7 ms** | **120.6 ms** | |
| **계란 1개당** | **6.5 ms** | **8.8 ms** | 시간당 환산 ≈ 40만 개 이상 처리 가능 (지연 KPI 는 CLAUDE.md 6장 미정) |

### 2-1. 배치 추론 개선 (2026-09-25, 2클래스 320px 모델, 판당 계란 12.1개)

| 분류기 실행 방식 | 분류 p50 | 판 합계 p50 | 계란당 p50 | 메모 |
|---|---:|---:|---:|---|
| ONNX Runtime CUDA, 계란별 배치 1 | 66.1 ms | 78.7 ms | 6.5 ms | 초기 측정 (256px 33클래스) |
| ORT CUDA, 동적 배치 | 92.4 ms | 103.4 ms | 8.2 ms | 판마다 계란 수가 달라 형태 변경 시 cuDNN 재탐색 → 악화 |
| ORT CUDA, 고정 배치 16 패딩 + HEURISTIC | 73.3 ms | 85.7 ms | 7.1 ms | 재탐색은 없어졌지만 CPU 전처리가 병목 |
| PyTorch FP16 배치, CPU 정규화 | 65.7 ms | 77.4 ms | 6.1 ms | |
| PyTorch FP16 배치, GPU 정규화 | 23.8 ms | 34.7 ms | 2.8 ms | CPU 는 리사이즈만, 정규화·채널 변환은 GPU |
| PyTorch FP16 배치, 리사이즈·정규화 모두 GPU | 15.3 ms | 25.9 ms | 2.1 ms | CPU 는 uint8 크롭 전송만 (`TimmClassifier._letterbox_gpu`) |
| **PyTorch FP16, 판 이미지 1회 전송 + GPU 크롭** | **13.0 ms** | **23.4 ms** | **1.9 ms** | `TimmClassifier.classify_boxes` — 파이프라인이 bbox 만 넘김, p95 계란당 9.2 ms |

결론: 병목은 모델이 아니라 **크롭 전처리와 전송** 이었고, 이를 GPU 로 옮겨 판당 분류 66 → 13 ms 로 줄였다. 이제 판 합계 23 ms 중 검출이 10 ms(순수 추론 1.9 ms + Ultralytics 전·후처리)로 가장 크다.

### 2-2. 검출 오버헤드 측정 (2026-09-25, test 판 60장)

| 입력 방식 | YOLO11n p50 | YOLO26n p50 |
|---|---:|---:|
| numpy 이미지 → `predict()` (현재) | 10.1 ms | 10.6 ms |
| 사전 letterbox 한 CUDA 텐서 → `predict()` | 11.4 ms | 13.4 ms |
| (참고) letterbox 전처리만 | 2.4 ms | 2.4 ms |

사전 텐서 입력으로는 **이득이 없다**(오히려 느림). 오버헤드는 전처리가 아니라 Ultralytics 예측기의 결과 객체 생성·후처리 쪽이며, 이를 줄이려면 예측기를 우회해 모델 forward + 자체 NMS(또는 YOLO26 의 NMS-free 출력)를 직접 다루거나 TensorRT 로 가야 한다. 현재 판당 23 ms(계란당 1.9 ms)는 어떤 현실적인 처리량 KPI 에도 충분하므로 이 최적화는 **장비 확정 후 TensorRT 작업과 함께** 진행한다. Jetson Orin NX 급에서는 대략 3~5배 느려질 것으로 예상하므로 현장 장비에서 재측정 필요.

## 3. 안전 동작 (구현·단위 테스트 완료, `tests/test_plc_and_pipeline.py`)

| 상황 | 동작 (configs/infer.yaml `safety`) | 알람 |
|---|---|---|
| 카메라 프레임 없음 | 라인 정지 요청 | E200 |
| 모델 로드 실패·추론 예외 | 전량 배출 | E100/E500 |
| PLC 통신 끊김 | 라인 정지 요청, 실패 시 로컬 기록 | E300 |
| 지연 초과 | 경고 알람 | W400 |

## 4. 남은 일 (장비 확정 후)

1. TensorRT FP16 변환 → INT8 은 정확도 비교 후 결정
2. 실제 PLC 드라이버(`src/plc/modbus_tcp.py`) 구현, 하트비트 끊김 시 PLC 측 안전 동작 래더 협의
3. 카메라 SDK 연동, 다중 프레임 투표(계란 회전 시)
4. 판정 로그 → 라벨링 → 재학습 루프
