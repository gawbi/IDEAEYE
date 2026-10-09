# 모니터링 웹서비스 실행 방법

설계: `docs/web_monitoring_design.md` · 코드: `src/webmon/` (서버), `src/telemetry/` (장비 업로더)

## 서버
```powershell
uvicorn src.webmon.app:app --host 0.0.0.0 --port 8000
```
- 대시보드: http://127.0.0.1:8000/ · API 문서: /docs · 상태: /healthz
- DB: `data/webmon.sqlite3` (환경변수 `EGGMON_DATABASE_URL` 로 PostgreSQL 등 교체)
- 관리자 토큰: 환경변수 `EGGMON_ADMIN_TOKEN` (기본 `admin-dev`, 운영 시 반드시 변경)

## 장비 등록과 설정 변경 (관리자)
```powershell
curl -X POST http://127.0.0.1:8000/api/v1/admin/devices -H "X-Admin-Token: admin-dev" -H "Content-Type: application/json" -d "{\"device_id\":101,\"name\":\"1호기\",\"site\":\"1농장\"}"
# 응답의 api_key 를 장비에 저장
curl -X PUT http://127.0.0.1:8000/api/v1/admin/devices/101/config -H "X-Admin-Token: admin-dev" -H "Content-Type: application/json" -d "{\"config\":{\"decision\":{\"crack_threshold\":0.45}}}"
```

## 가상 장비로 데모
```powershell
python scripts\simulate_device.py --server http://127.0.0.1:8000 --device-id 101 --days 3 --live 60
```

## 장비(Jetson)에서
```python
from src.telemetry.uploader import Aggregator, HttpTransport, LocalQueue, Uploader
up = Uploader(HttpTransport("https://서버", device_id=101, api_key="..."), LocalQueue("/var/lib/eggmon/queue.sqlite3"), on_config=apply_config)
# 판정 루프에서: st = ag.add(reject, severity); 주기적으로 up.enqueue_counts(st); up.send_pending(); up.heartbeat(...); up.poll_config()
```
- 통신이 끊겨도 큐에 보관되고 복구 후 순서대로 재전송(멱등). 전송량은 일 집계 1 KB 미만.
- KS 송출 데이터: `GET /api/v1/export/ks?device_id=101&date_from=2026-10-01` → CSV(검출기 번호, 처리란 수, 파각란 수, 설정 시간).

## 남은 작업
- 로그인(대시보드 접근 제어), HTTPS 리버스 프록시, 서버 호스팅 위치 결정
- (선택) 오판 의심 이미지 업로드·검토 화면, MQTT 어댑터(LTE Cat.M1)
