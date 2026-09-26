# AI 기반 파각 검출 및 검란 시스템 — 프로젝트 뼈대

프로젝트 컨텍스트는 [CLAUDE.md](CLAUDE.md), 단계별 작업은 [PROMPTS.md](PROMPTS.md) 참고.

```
configs/        data.yaml(경로·좌표계·XML 스키마) split.yaml infer.yaml roboflow.yaml aihub_official_counts.yaml
src/common/     config.py        YAML 로더
src/datasets/   aihub_xml.py     AI Hub XML 파서 (태그 경로·좌표계 설정 가능)
                coords.py        bbox 좌표계 변환
                split.py         group split + 누수 검사
                audit.py         Phase 1 데이터 감사 (시각화·통계·매칭 키)
                roboflow_yolo.py Roboflow YOLO export 분석 (클래스 수·source 누수·라벨 형식)
                fake_aihub.py    합성 데이터 생성기 (단위 테스트 전용, 실데이터 대체 아님)
src/eval/       metrics.py       소분류별 P/R, 운영점 계산
                tracker.py       experiments/ 기록 (config·seed·git hash·지표)
src/infer/      pipeline.py      추론 파이프라인 뼈대
src/plc/        base.py mock.py  PLC 인터페이스 + mock
src/train/      classifier.py    크롭 분류기 학습·평가 (전처리·손실 ablation, 운영점, 축별 지표)
src/models/     patchcore.py     PatchCore-lite 이상탐지
scripts/        make_split.py → make_crops.py → export_det.py   데이터 준비 (Phase 2)
                train_det.py / train_cls.py / train_patchcore.py / candling_threshold.py   학습 (Phase 3·5)
                fn_analysis.py / ablation_table.py / candling_confusion.py   분석 (Phase 4·5)
                export_and_benchmark.py   ONNX + 지연 벤치마크 (Phase 7)
                own_domain_adapt.py       자사 데이터 도메인 적응 (Phase 6, 데이터 입고 시)
                run_queue.py + configs/queue_*.yaml   GPU 작업 순차 큐 / auto_next.py 결과 기반 후속 실험
                watch_queue.py, watch_experiments.py, wait_for.py   완료 감시 (백그라운드 알림용)
                collect_results.py → experiments/RESULTS.md, contact_sheets.py → outputs/review/, label_quality.py
                ingest_roboflow_zip.py / download_roboflow.py   보조 데이터
docs/           final_summary.md ← 먼저 읽기. phase1~7 문서, TODO.md, datasets.md, inference_pipeline.md, plc_interface.md
검수/           사람이 확인할 자료 (격자 이미지, 좌표계, 결과 문서)
tests/          단위 테스트 56개 (python -m pytest)
```

## 실험 재실행

```powershell
python scripts\run_queue.py configs\queue_all.yaml      # Phase 3~7 전체 (약 10시간, RTX 4060)
python scripts\train_cls.py --config exp_cls_mono_crackbin   # 단일 실험
python scripts\collect_results.py; python scripts\ablation_table.py
```

## 데이터 폴더 규칙

```
data/aihub/                 AI Hub 계란 데이터 (jpg + xml)            ← 1순위, KPI 산출용
data/own/                   자사 라인 수집 데이터
data/roboflow/<slug>/       Roboflow export (train|valid|test/{images,labels}, data.yaml, summary.json)
data/roboflow/_zips/        반입한 원본 zip 보관
data/roboflow/manifest.json 반입 데이터셋 요약 (docs/datasets.md 의 원천)
data/crops/  data/splits/   파생물
```

Roboflow zip 이 새로 생기면 한 줄로 반입한다:

```powershell
python scripts\ingest_roboflow_zip.py "<파일>.zip"
```

## 빠른 시작

```powershell
python -m pip install -r requirements.txt
python -m pytest
python -m src.datasets.fake_aihub data\fake_aihub          # 합성 데이터
python -m src.datasets.audit --root data\fake_aihub --out outputs\audit_fake
$env:ROBOFLOW_API_KEY = "..." ; python scripts\download_roboflow.py
```
