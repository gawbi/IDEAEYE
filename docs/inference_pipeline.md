# 추론 파이프라인 설계 (Phase 7 선행 설계)

구현 뼈대: `src/infer/pipeline.py` · 설정: `configs/infer.yaml` · 테스트: `tests/test_plc_and_pipeline.py`

## 1. 단계

| # | 단계 | 입력 → 출력 | 지연 측정 키 | 비고 |
|---|---|---|---|---|
| 0 | 캡처 | 카메라 → 판 이미지 (BGR) | (캡처 모듈에서 측정) | MONO/COLOR 두 대면 프레임 동기화 필요 |
| 1 | 계란 검출 | 판 이미지 → bbox 목록 | `detect` | YOLO11 (TensorRT FP16) |
| 2 | 크롭 분류 | bbox 여백 10% 크롭 → (소분류, 점수) | `classify` | 배치 추론으로 계란 수만큼 반복 비용 절감 |
| 3 | 판정 | (라벨, 점수) → 배출 여부 | (2에 포함) | `DecisionRule`: 클래스별 임계값, 운영점은 파각 Recall 목표로 결정 |
| 4 | PLC 신호 | 배출 목록 → `send_reject` | `plc` | 판 시퀀스 + 계란 인덱스 → 배출 기구 위치로 매핑 (PLC 기종 확정 후) |
| 5 | 로그 | 판 결과 → JSONL (+이미지) | (별도) | `outputs/infer_logs/`, 재학습 데이터 원천 |

`TrayResult.timings_ms` 에 단계별 시간이 기록되며 `total` 이 KPI(계란 1개당 지연) 계산의 기준이다.
판 단위 총 지연을 판 내 계란 수로 나눠 계란 1개당 지연을 보고한다.

## 2. 안전 동작 (configs/infer.yaml `safety`)

| 상황 | 감지 방법 | 기본 정책 | 알람 코드 |
|---|---|---|---|
| 모델 로드 실패 / 추론 예외 | 파이프라인 예외 | `reject_all` (전량 배출) | E100 / E500 |
| 카메라 프레임 없음 | `image is None`, 타임아웃 | `stop_line` | E200 |
| PLC 통신 끊김 | `PLCError` | `stop_line` (PLC 자체가 죽었으면 로컬 기록만) | E300 |
| 지연 초과 | `total > max_latency_ms` | 경고 알람만 | W400 |

원칙: **판정 불가 상태에서 정상 배출은 절대 하지 않는다.** 정책은 현장 협의 후 확정 (전량 배출 vs 라인 정지는 라인 구조에 따라 다름).
하트비트: 추론 서비스가 `heartbeat_interval_s` 마다 `heartbeat()` 를 보내고, PLC 는 일정 시간 끊기면 스스로 안전 동작을 취하도록 래더를 구성한다 (소프트웨어 죽음 대비).

## 3. 다중 프레임 투표 (선택)

계란이 라인에서 회전한다면(CLAUDE.md 2장 확인 필요) 같은 계란의 N 프레임 결과를 모아 투표한다.
현 뼈대는 프레임 단위 판정만 하므로, 추적기(ByteTrack 등)를 단계 1과 2 사이에 두고 계란 ID 별 점수 이력을 유지하는 계층을 추가한다.

## 4. 벤치마크

`scripts/benchmark_latency.py`(Phase 7 작성 예정): 저장된 판 이미지 100장을 파이프라인에 넣고 단계별 p50/p95/max 를 표로 출력. 대상 장비(Jetson / IPC+RTX)별로 실행 결과를 `experiments/` 에 기록한다.

## 5. 미확정 항목 (CLAUDE.md 2장에 채워야 함)

- 카메라 대수·해상도·fps, MONO/COLOR 동시 촬영 여부
- 처리량과 허용 지연 → `max_latency_ms`
- 배출 기구의 위치 지정 방식 → `RejectCommand` 필드 확장
- 추론 장비 → TensorRT INT8 채택 여부
