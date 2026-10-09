# 파각란검출기 관리·모니터링 웹서비스 설계 (필수-4)

작성 2026-10-09 · 상태: 설계안(구현 전 검토용). 근거: KS B 7956-4 4.3.2 송출 데이터, 교수님 답변(WiFi 기본, LTE Cat.M1 옵션), 현재 추론 파이프라인의 판정 로그.

## 1. 목표와 범위

- 검출기(Jetson)가 **무선으로 서버에 상태·집계·판정 로그를 올리고**, 관리자가 웹 화면에서 장비 상태와 일별 실적을 보며, 필요 시 설정(임계값 등)을 내려보낼 수 있다.
- KS 송출 데이터 4종을 반드시 포함: 검출기 ID(농장 단위 정수), 일별 처리란 개수, 일별 파각란 개수, 설정 시각(YYYYMMDDhhmmss).
- 1차 범위(캡스톤): 서버 API + 대시보드 + 장비 쪽 업로더(시뮬레이터 포함). LTE Cat.M1 모듈 실장(선택-1)은 통신 어댑터만 교체하면 되도록 분리.

## 2. 제약에서 나오는 설계 결정

| 제약 | 결정 |
|---|---|
| LTE Cat.M1 은 저대역폭(수십 kbps 급)·종량제 | **집계값 우선 전송**(일별/시간별 카운트, 수 KB). 계란별 로그는 배치 압축 전송(선택), 이미지는 **오판 의심 사례만** 축소본 전송(선택) |
| 통신 끊김 잦음 | 장비 쪽 **로컬 큐(SQLite)** 에 먼저 저장 → 연결되면 순서대로 재전송(멱등 키로 중복 방지) |
| 장비 시계 신뢰 불가 | 서버 수신 시각과 장비 시각을 둘 다 저장. KS "설정 시각"은 장비 설정 변경 시각으로 기록 |
| 여러 검출기 | device_id 로 분리, 농장(site) 단위 그룹 |
| 보안 | 장비별 토큰(API key) + HTTPS. 대시보드는 로그인 |

## 3. 데이터 모델 (서버, SQLite → 필요 시 PostgreSQL)

| 테이블 | 주요 열 |
|---|---|
| `devices` | device_id(정수, KS 검출기 번호), name, site, api_key_hash, model_version, config_json, created_at, last_seen_at |
| `heartbeats` | device_id, device_ts, server_ts, status(ok/alarm), cpu/gpu/temp, queue_len, fps |
| `daily_counts` | device_id, date, processed_count, cracked_count, severity_1/2/3_count, updated_ts (**KS 송출 항목**) |
| `hourly_counts` | device_id, hour_ts, processed, cracked (추이 그래프용) |
| `events` | device_id, device_ts, type(alarm/config/start/stop), code(E100…W400), message |
| `egg_logs` (선택) | device_id, egg_uid, device_ts, cam, roi, scores_json(3시점), fused_score, reject, severity, image_key |
| `images` (선택) | image_key, device_id, egg_uid, path, size |
| `configs` | device_id, version, json(임계값·투표 규칙·safety), set_ts(**KS 설정 시각**), applied_ts |

## 4. API (REST, JSON, 장비 → 서버는 모두 멱등)

| 메서드 | 경로 | 용도 |
|---|---|---|
| POST | `/api/v1/devices/{id}/heartbeat` | 30~60초 주기 상태 |
| POST | `/api/v1/devices/{id}/counts` | 일별/시간별 집계 upsert (`{date, processed, cracked, severity:{1,2,3}}`) |
| POST | `/api/v1/devices/{id}/events` | 알람·상태 변화 (배열) |
| POST | `/api/v1/devices/{id}/eggs` | 계란 판정 로그 배치 (gzip JSON, 선택) |
| POST | `/api/v1/devices/{id}/images` | 오판 의심 이미지 (multipart, 선택) |
| GET | `/api/v1/devices/{id}/config` | 적용할 설정 조회 (장비가 폴링; 변경 시 version 증가) |
| GET | `/api/v1/devices`, `/api/v1/devices/{id}/summary?from&to` | 대시보드용 |
| GET | `/api/v1/export/ks?device_id&date` | KS 송출 형식(검출기 번호, 처리란 수, 파각란 수, 설정 시간) |

인증: `Authorization: Bearer <device api key>`. 서버 → 장비 푸시는 두지 않고 **장비 폴링**으로 단순화(LTE 환경에서 NAT·연결 유지 문제 회피). MQTT 는 Cat.M1 모듈 채택 시 어댑터로 추가(토픽 `egg/{device_id}/counts` 등, 페이로드 동일).

## 5. 장비 쪽 업로더 (Jetson 에서 실행, `src/telemetry/`)

- 파이프라인의 `JudgmentLogger` 가 쓰는 JSONL 을 읽어 **분 단위 집계**를 만들고 로컬 SQLite 큐에 적재.
- 전송 루프: 연결 가능하면 큐 순서대로 전송, 실패 시 지수 백오프. 하트비트는 별도 주기.
- 전송 데이터 크기 목표: 하루 집계 1 KB 미만, 하트비트 200 B, 계란 로그 선택 시 하루 수백 KB(gzip).
- 설정 폴링: 10분마다 `/config` 조회 → 버전이 바뀌면 `configs/infer.yaml` 의 decision 부분을 갱신하고 `applied` 이벤트 전송.

## 6. 대시보드 화면

1. **장비 목록**: 상태(온라인/오프라인/알람), 마지막 수신, 오늘 처리란·파각란·파각률.
2. **장비 상세**: 시간별 처리·파각 추이 그래프, 심각도 분포, 알람 이력, 현재 설정과 버전, 설정 변경 폼(임계값·투표 규칙).
3. **오판 검토(선택)**: 업로드된 의심 이미지와 점수를 보고 라벨을 달아 재학습 데이터로 보냄.
4. **KS 송출 내보내기**: 기간·장비 선택 → CSV.

기술: FastAPI + SQLite(SQLAlchemy) + Jinja2 템플릿 + 경량 차트(Chart.js CDN). 별도 프론트 프레임워크 없이 시작해 범위를 작게 유지.

## 7. 구현 순서 (각 단계마다 테스트 포함)

1. 서버 API + DB + 가상 장비 시뮬레이터(집계·하트비트·이벤트 전송) → 단위 테스트
2. 대시보드(목록·상세·KS 내보내기)
3. 장비 업로더(JSONL → 집계 → 큐 → 전송, 오프라인 재전송 테스트)
4. 설정 폴링·적용
5. (선택) 계란 로그·이미지 업로드, MQTT 어댑터

## 8. 미정·질문

- 농장/검출기 ID 체계(KS 는 농장 단위 고유 정수) — 기업 확인.
- 서버 호스팅 위치(학교 서버 / 클라우드 / 기업) 와 도메인·인증서.
- LTE Cat.M1 모듈 종류와 요금제 → 전송량 상한 결정.
- 기존 운영 소프트웨어(시제품 모니터의 로그인 화면)와의 관계: 별도 서비스인지, 그 UI 에 통합인지.
