# Phase 6 자사 라인 도메인 적응 — 절차 (데이터 입고 대기)

2026-09-24 기준 `data/own/` 에 데이터가 없어 실행하지 못했다. 데이터가 들어오면 아래 순서대로 한 번에 돌아간다.

## 1. 데이터 준비

- 판 이미지 + AI Hub 형식 XML 이면: `configs/data.yaml` 의 `own_root` 에 두고 `python scripts/make_crops.py`(경로만 바꿔) 로 크롭·index 생성.
- 크롭 단위 라벨이면: `data/own/index.csv` 에 `crop_path, cls(state/state_detail 코드), split(train|test), light` 를 채운다.
- **자사 test 셋은 처음 한 번 고정하고 이후 절대 바꾸지 않는다** (CLAUDE.md 설계 원칙 5). 판·촬영일 단위로 나눈다.

## 2. 실행

```powershell
python scripts/own_domain_adapt.py --base latest:cls_mono_effb0 --sizes 100 300 1000 3000
python scripts/own_domain_adapt.py --base latest:cls_mono_effb0 --sizes 100 300 1000 3000 --flatfield
```

산출: `experiments/<stamp>_own_adapt_MONO[/_ff]/learning_curve.json` — zero-shot(0장) → 100 → 300 → 1000 → … 의 macro recall. 곡선이 평평해지는 지점이 추가 라벨링 필요량의 근거.

## 3. 판단 기준

| 결과 | 해석 |
|---|---|
| zero-shot 이 AI Hub test 대비 크게 낮음 | 도메인 이동 큼 → 조명·카메라 차이. flat-field 결과와 비교 |
| flat-field 로 zero-shot 상승 | 조명 불균일이 주요 원인 → 라인에 확산판 참조 영상 촬영 절차 추가 |
| 300장에서 수렴 | 라벨링 300~500장이면 충분 |
| 3000장에서도 상승 중 | 결함 유형이 AI Hub 와 다름 → 소분류별로 부족한 클래스 확인 |

flat-field 구현: `src/datasets/flatfield.py` (참조 영상이 있으면 정석 보정, 없으면 저주파 조명장 추정).
