# 로보플로우 보조 데이터 sanity check (자동 생성: scripts/roboflow_sanity.py)

우리 모델을 촬영 환경이 전혀 다른 공개 사진에 그대로 적용한 결과. **KPI 가 아니며** 도메인 이동의 크기를 가늠하는 용도. 임계값은 AI Hub test 에서 파각 Recall 95% 인 값을 그대로 사용.

| 데이터셋 | 모델 | 크랙 n | 정상 n | AUROC(크랙 vs 정상) | 크랙 Recall@AI Hub 임계값 | 정상 FPR@같은 임계값 | 이 데이터에서 Recall 95% 맞출 때 FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| cracked-eggs | MONO 2클래스 | 1,161 | 989 | 0.901 | 0.991 | 0.757 | 0.388 |
| cracked-eggs | COLOR 5클래스 | 1,161 | 989 | 0.861 | 0.968 | 0.903 | 0.836 |
| egg-crack-detection-eu65a | MONO 2클래스 | 1,515 | 0 | nan | 1.000 | nan | nan |
| egg-crack-detection-eu65a | COLOR 5클래스 | 1,515 | 0 | nan | 0.972 | nan | nan |
| egg-quality-amdky | - | 0 | 359 | 크랙 라벨 없음 → 건너뜀 | | | |

해석: AUROC 가 0.5 근처면 도메인 이동이 너무 커서 전혀 전이되지 않는 것, 0.8 이상이면 특징은 전이되지만 임계값 재조정이 필요한 것. 이 사진들은 스톡 사진·그레이스케일 변환·배경 제거가 섞여 있어 자사 라인보다 훨씬 먼 도메인이다.