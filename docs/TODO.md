# TODO — 자율 진행 기준 목록

Claude 는 깨어날 때마다 이 파일을 읽고 위에서부터 "할 수 있는 첫 항목"을 진행한다. 상태: `[ ]` 대기 · `[~]` 진행 중 · `[x]` 완료 · `[!]` 사용자 입력 필요.

## A. 완료 (2026-09-23 ~ 25) — 요약 `docs/final_summary.md`
- [x] Phase 1 데이터 감사, Phase 2 분할·크롭
- [x] Phase 3 A/B/C 비교 (검출기 0.670/0.582, 33클래스 분류기, PatchCore 제외)
- [x] Phase 4 미검출 분석 + ablation (320px −2.1p, focal −6.2p, CLAHE 악화, 조합 −7.8p) + 후속(6클래스 X, **2클래스 헤드 FPR@95 9.6%**)
- [x] Phase 5 검란 (분류기 0.974/0.979, 통합 33클래스 0.996/0.994, 전처리 원본 최선, 혈반↔먹통 혼동 미미)
- [x] Phase 7 ONNX·벤치마크·안전 동작 (TensorRT·실 PLC 는 장비 대기)
- [x] 3클래스(정상/크랙/기타) 분류기 + 배출 정책 시뮬레이션 (`docs/reject_policy.md`): 크랙만 기준 12.3%, 모든 결함 95% 배출 시 정상 오배출 33% → 정책 결정 필요
- [x] 파형크랙(2/7) FN 격자 (`검수/06_미검출_파형크랙/`): 미검출은 단일 시점에서 균열이 안 보임 → 다중 프레임 촬영 권고
- [x] 배치 추론: ORT 동적 배치 악화 원인(형태 변경 시 cuDNN 재탐색) 규명, PyTorch FP16 + GPU 정규화로 분류 66 → 24 ms/판, 계란당 2.8 ms
- [x] 문서: phase1/3/4/5/6/7, final_summary, reject_policy, RESULTS.md, CLAUDE.md 진행 현황, 검수 폴더 01~08
- [x] 2026-09-26: 로보플로우 sanity check(AUROC 0.901, 임계값 재조정 필요), 데모 추론 도구(`scripts/demo_infer.py`, 검수/07), 데이터 수집 요청서(`docs/data_collection_request.md`, 검수/08), git 초기화(휴대용 MinGit `C:\tools\mingit`, 사용자 PATH 등록)

## A-2. 2026-10-09 — 기업 정보 반영 후 자율 작업 (기기 영상 입고 전 준비)
- [x] 현장 정보 정리 `docs/field_info.md` (카메라 ELP IMX678 4K60 화각 94°, Jetson Orin Nano, 카메라 2대×3자리, 120°×3회 촬영, KS B 7956-4 미검출 7%, 종란 검란은 11월, 영상 10/11~, 라벨링 직접)
- [x] 1. 영상 → 크롭 파이프라인 `scripts/ingest_video.py` (화면 분리, 투광 빛 누적 ROI 자동 검출+k-means 분할, 카메라별 정지 구간, 계란 유무) — 샘플 57초 → 72크롭, 검수/09. AI Hub 모델 zero-shot 은 전부 크랙 판정(0.96) → 무의미, 재학습 필수
- [x] 2. 라벨링 체계: `docs/labeling_guide.md`, 키보드 도구 `scripts/label_crops.py`, 통합 `scripts/own_build_index.py`(3시점 egg_id 그룹 분할)
- [x] 4. 3시점 투표 `src/infer/voting.py` (max/mean/kofn/hybrid, 타임아웃, 독립 가정 시뮬레이션: kofn(2) 최종 Recall 95% 에 시점 Recall ≥ 0.88 필요), configs/infer.yaml multiview
- [x] 3. 파각 정도 프로토타입 `configs/exp_sev_mono.yaml` (AI Hub 소분류 약한 라벨 3등급) — test 정확도 0.820, macro Recall 0.787 (경미 0.62 / 보통 0.85 / 심각 0.89), ±1등급 허용 0.991, 극단 혼동 3~4%. 결론: 심각도 헤드 가능, 자사 severity 라벨로 재학습 필요. 상세 `docs/severity_prototype.md`. (평가 중 DataLoader 워커 페이징 파일 부족으로 멈춤 → workers=4 로 평가 재실행; 이 PC 는 workers ≤ 4 권고)
- [x] 소프트웨어 아키텍처 구조도 v2 (로고 블록 + 프로토콜 화살표 + 7단계 워크플로, 구현 상태 범례) `검수/10_웹모니터링/architecture.png`, `docs/web_architecture.png`
- [x] 5. 웹 모니터링: 설계 `docs/web_monitoring_design.md`, 서버 `src/webmon/`(FastAPI+SQLite, KS 송출 CSV, 대시보드), 업로더 `src/telemetry/uploader.py`(집계·오프라인 큐·백오프·설정 폴링), 시뮬레이터 `scripts/simulate_device.py`, 실행법 `docs/webmon_howto.md`. 테스트 6개 통과
- [x] 6. Jetson 준비 `deploy/jetson/` (ONNX 고정 배치 내보내기, setup/build_trt/bench 스크립트, README)
- [x] 7. 문서: KS 내부 시험 계획 `docs/ks_test_plan.md`, 데이터 요청서 4-1절 추가(원본 프레임·회전 신호·두 난각색)

## B. 다음에 Claude 가 할 일 (사용자 결정 후 또는 선택)
- [ ] (기기 영상 입고 시) ingest_video → 라벨링(팀) → own_build_index → `configs/exp_own_*.yaml` 로 AI Hub 가중치 초기화 미세조정 → 3시점 투표 운영점 → KS 시험 계획으로 평가
- [ ] (파각 정도 결과 후) 심각도 모델 + 빛샘 영역 물리 지표(길이·면적) 결합 설계
- [ ] 웹서비스: 로그인, HTTPS, (선택) 이미지 업로드·검토 화면, MQTT 어댑터
- [ ] 파이프라인에 고정 ROI 모드(검출기 생략) + 3시점 EggAggregator 통합, Jetson 용 TrtClassifier 어댑터
- [ ] (정책 확정 후) 정책 기준 운영점 최종표 + 판정 규칙을 `configs/infer.yaml decision` 에 반영, 2클래스+33클래스 이중 헤드 파이프라인 어댑터
- [x] 리사이즈까지 GPU(torch interpolate): 분류 판당 15.3 ms, 판 합계 25.9 ms, 계란당 p50 2.1 ms
- [x] (선택) YOLO26n 검출기 비교(MONO): mAP@0.5 0.675 vs 0.670, 위치 Recall 0.985 vs 0.984, Precision 0.970 vs 0.956 → 동등(정밀도 소폭 우세, NMS-free)
- [x] (선택) 2클래스 EfficientNet-B2: FPR@95 8.9% (B0 9.6%), 지연 1.5배 → 소폭 개선
- [x] 크롭 전송 판 단위 일괄화(`TimmClassifier.classify_boxes`): 분류 판당 13.0 ms, 판 합계 23.4 ms, 계란당 p50 1.9 ms
- [x] (선택) 검출 오버헤드 측정: 사전 텐서 입력은 이득 없음(11.4 vs 10.1 ms), YOLO26n 도 동일(10.6 ms). 오버헤드는 Ultralytics 예측기 후처리 → 예측기 우회 또는 TensorRT 필요. 판당 23 ms 로 충분해 장비 확정 후로 미룸 384px 크롭은 원본(~250×320)을 확대할 뿐이라 제외
- [ ] (선택) 2/7 라벨 정제(육안 검토 결과 반영) 후 2클래스 재학습
- [ ] (데이터 입고 시) Phase 6: `scripts/own_domain_adapt.py`

## C. 사용자 입력 필요 `[!]`
- [!] **배출 정책**: 어떤 결함을 배출하는가 (크랙만 / 크랙+혈반 / 탈색·이물질 포함 전부)
- [!] CLAUDE.md 2장 현장 조건, 6장 KPI 목표치 (파각 Recall, 허용 오배출률, 지연), 계란 회전 여부(다중 프레임 가능성)
- [!] 검란 범위 (식용란 / 종란 / 둘 다), 먹통 배출 여부
- [!] AI Hub 데이터 설명서 코드표 → `configs/data.yaml` class_map 확정 (크랙 소분류 2/1~2/7, 정상·탈색·이물질·기형 소분류 이름)
- [!] `검수/06` 파형크랙 미검출 격자 육안 확인 결과
- [!] (선택) AI Hub 제공 DETR 모델 파일 → `models/aihub/`
- [!] 자사 라인 데이터 (`data/own/`, 세척/비세척을 같은 세션에 혼합 촬영), 추론 장비·PLC 기종
- [x] git 저장소 초기화 (2026-09-26 완료, 커밋 2건)

## D. 규칙
- GPU 작업은 반드시 하나의 큐(`scripts/run_queue.py`)로만 실행. 병렬 GPU 사용 금지 (cuDNN 오류 경험).
- 큐 완료 대기는 프로세스 조회 대신 상태 파일 폴링(`wait_for.py --status-file`, `watch_queue.py`, `watch_experiments.py`). Get-CimInstance 는 이 PC 에서 분 단위로 느림.
- 실패한 작업은 로그(outputs/logs/<name>.log) 확인 후 원인 수정하고 재실행. 원인 불명이면 TODO 에 기록.
- 결과 수치는 코드 가설(configs/class_codes.yaml) 기반임을 문서에 명시.
