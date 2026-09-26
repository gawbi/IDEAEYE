# Phase 1 데이터 감사 절차 (AI Hub 계란 데이터 71504)

실행 순서와 확인 항목. 실데이터를 `data/aihub/` 에 풀어둔 뒤 진행한다.

## 0. 스키마 (2026-09-23 실데이터로 확인 완료)

- 다운로드본: `C:\Users\user\Downloads\248.계란 데이터\01-1.정식개방데이터\{Training,Validation}\{01.원천데이터,02.라벨링데이터}\` zip 12개(이미지 92GB, 라벨 0.4GB).
- 라벨만 `C:\aihub_egg\labels\{Training,Validation}\{MONO,COLOR,STORAGE}\` 에 풀어 둠. 이미지는 풀지 않고 zip 에서 직접 읽는다 (`ImageSource`).
- XML: 평면 `<annotation>` 에 메타 태그(category, regional_code, varieties, wash, classification, days, type, create_date, date, temperature, humidity, file_name, fresh_day, breeding_env), `<size>`, `<object_count>`, 객체는 `<bndbox>`(seq, object, state, state_detail, x_min, y_min, x_max, y_max).
- **state / state_detail / wash / varieties / breeding_env / type 은 숫자 코드.** AI Hub 데이터 설명서의 코드표를 `configs/data.yaml` 의 `class_map` 에 채우기 전까지는 코드 그대로 집계한다. 추측 금지.
- 모델 설명서에 따르면 MONO 에는 정상~기형 6종 외 **N/A 클래스**가 있음 → 코드 하나는 N/A 일 가능성.

## 1. 실행

```powershell
python -m src.datasets.audit --n-vis 24          # 실데이터 (라벨 13.7만 개, 병렬 파싱 약 1.5분 + 표 생성)
# 결과 캐시: outputs/audit/records.pkl (객체 단위 124만 행, 287MB) → Phase 2 분할·크롭은 재파싱 없이 이 파일을 읽는다
# 합성 데이터로 동작 확인:
python -m src.datasets.fake_aihub data\fake_aihub
python -m src.datasets.audit --root data\fake_aihub --out outputs\audit_fake
```

## 2. 산출물 (outputs/audit/)

| 파일 | 내용 |
|---|---|
| `bbox_<light>_<tray>.jpg` | 왼쪽 = top_left 해석, 오른쪽 = bottom_left 해석. 파란 점 = 원본 (x_min, y_min) 위치 |
| `origin_evidence.csv` | 판별 bbox 내부/외부 전경 비율 차이, 범위 밖 박스 수 |
| `official_vs_actual.csv` | CLAUDE.md 공식 수치 vs 실측, `diff` 열 |
| `table_state.csv`, `table_state_detail.csv` | MONO/COLOR 별 대분류·소분류 개수 |
| `table_tray/date/region_code/washed/breed.csv` | 분포표 |
| `report.md`, `report.json` | 종합 보고 (매칭 키 후보 포함) |

## 3. 좌표계 확정 기준

- 시각화에서 **박스가 계란에 정확히 맞는 쪽**이 정답. `origin_evidence` 의 `prefer` 다수결은 보조 근거일 뿐이다.
- 확정되면 `configs/data.yaml` 의 `coords.origin` 을 바꾸고 `verified: true` 로 표시, CLAUDE.md 진행 현황에 근거(이미지 파일명)를 적는다.
- 판 이미지가 상하 대칭에 가까우면 수치 근거의 `margin` 이 작다. 이 경우 크랙 표시 등 비대칭 단서를 눈으로 확인한다.

## 4. 완료 기준 (PROMPTS.md Phase 1)

- [ ] 좌표계 확정 + 근거 이미지
- [ ] 소분류 통계표와 공식 수치 차이 설명
- [ ] MONO↔COLOR 매칭 키 확정 (없으면 "매칭 불가" 로 기록하고 Phase 4 융합 실험 제외)
