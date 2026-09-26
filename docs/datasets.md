# 데이터셋 인벤토리 (보조 공개 데이터)

`scripts/ingest_roboflow_zip.py` 가 자동 생성. **여기서 나온 수치는 KPI 로 쓰지 않는다** (CLAUDE.md 3.2).
재분할 시에는 반드시 `source` 단위 group split 을 쓴다 (증강본이 같은 source 를 공유).

## cracked-eggs

- 출처: https://universe.roboflow.com/crackedeggs/cracked-eggs/dataset/8 (v8, CC BY 4.0)
- 클래스: cracked-egg, normal-egg
- 라벨 형식: polygon (polygon 이면 YOLO segment 형식, detect 학습 시 bbox 로 자동 환산됨)
- 이미지 1790장 / 원본(source) 501장
- 전처리: Auto-orientation of pixel data (with EXIF-orientation stripping); Resize to 640x640 (Stretch); Grayscale (CRT phosphor)
- 증강(x3): 50% probability of horizontal flip; 50% probability of vertical flip; Random rotation of between -45 and +45 degrees; Salt and pepper noise was applied to 1.37 percent of pixels
- 분할 간 source 이름 중복 (train∩val): 20건 — 일반 이름(동명이인 가능) 14건, 이름으로 확정 6건 예: LINE_ALBUM_-_240410_10, LINE_ALBUM_-_240410_2, LINE_ALBUM_-_240410_6. 재분할 시 같은 이름은 같은 그룹으로 묶을 것(보수적)

| split | images | sources | labels | missing | empty | cracked-egg | normal-egg | boxes/img (min/mean/max) | sizes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| train | 1650 | 383 | 1650 | 0 | 6 | 1185 | 1115 | 0/1.39/13 | 640x640×1650 |
| val | 140 | 138 | 140 | 0 | 1 | 118 | 125 | 0/1.74/12 | 640x640×140 |

## egg-crack-detection-eu65a

- 출처: https://universe.roboflow.com/innodatatics-yyp09/egg-crack-detection-eu65a/dataset/1 (v1, CC BY 4.0)
- 클래스: Brown cracked egg, White cracked egg
- 라벨 형식: bbox (polygon 이면 YOLO segment 형식, detect 학습 시 bbox 로 자동 환산됨)
- 이미지 3154장 / 원본(source) 1054장
- 전처리: Auto-orientation of pixel data (with EXIF-orientation stripping); Resize to 640x640 (Fit within)
- 증강(x3): 50% probability of horizontal flip; 50% probability of vertical flip; Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise; Random brigthness adjustment of between 0 and +13 percent
- 분할 간 source 중복: 없음

| split | images | sources | labels | missing | empty | Brown cracked egg | White cracked egg | boxes/img (min/mean/max) | sizes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| train | 3154 | 1054 | 3154 | 0 | 0 | 1743 | 1522 | 1/1.04/4 | 480x640×1262, 640x480×939, 640x640×160 |

## egg-quality-amdky

- 출처: https://universe.roboflow.com/egg-quality-grading-rceef/egg-quality-amdky/dataset/1 (v1, CC BY 4.0)
- 클래스: blod stained, blood stained, brown egg, calcium deposit, dirt eggs, white eggs
- 라벨 형식: bbox (polygon 이면 YOLO segment 형식, detect 학습 시 bbox 로 자동 환산됨)
- 이미지 124장 / 원본(source) 118장
- 전처리: Auto-orientation of pixel data (with EXIF-orientation stripping); Resize to 640x640 (Stretch)
- 분할 간 source 이름 중복 (train∩val): 3건 — 일반 이름(동명이인 가능) 3건, 이름으로 확정 0건. 재분할 시 같은 이름은 같은 그룹으로 묶을 것(보수적)
- 분할 간 source 이름 중복 (val∩test): 1건 — 일반 이름(동명이인 가능) 1건, 이름으로 확정 0건. 재분할 시 같은 이름은 같은 그룹으로 묶을 것(보수적)

| split | images | sources | labels | missing | empty | blod stained | blood stained | brown egg | calcium deposit | dirt eggs | white eggs | boxes/img (min/mean/max) | sizes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| train | 87 | 85 | 87 | 0 | 0 | 1 | 18 | 86 | 17 | 38 | 95 | 1/2.93/11 | 640x640×87 |
| val | 25 | 25 | 25 | 0 | 0 | 1 | 6 | 32 | 3 | 24 | 11 | 1/3.08/12 | 640x640×25 |
| test | 12 | 12 | 12 | 0 | 0 | 0 | 1 | 10 | 3 | 6 | 7 | 1/2.25/4 | 640x640×12 |
