# PLC 배출 연동 인터페이스

구현: `src/plc/base.py` (추상), `src/plc/mock.py` (mock) · 실제 구현은 PLC 기종 확정 후 `src/plc/modbus_tcp.py` 등으로 추가.

## 1. 계약 (PLCInterface)

| 메서드 | 의미 | 실패 시 |
|---|---|---|
| `connect()` / `disconnect()` | 세션 수립·종료 | `PLCError` |
| `is_connected` | 링크 상태 | - |
| `send_reject(RejectCommand)` | 계란 1개 배출 | `PLCError` → 파이프라인이 `on_plc_lost` 정책 적용 |
| `reject_all(tray_seq, reason)` | 판 전량 배출 (안전 동작) | `PLCError` |
| `stop_line(reason)` | 라인 정지 요청 (안전 동작) | `PLCError` |
| `heartbeat()` | 주기적 생존 신호 | `PLCError` |
| `alarm(code, message)` | 알람 통보. 연결과 무관하게 로컬 기록 보장 | 예외 없음 |
| `reset_alarm()` | 알람 해제 | - |

`RejectCommand`: `tray_seq`, `egg_index`, `reason`(소분류), `score`, `extra`.
배출 기구가 "판 내 행·열" 또는 "레인 번호 + 타이밍" 중 무엇을 요구하는지에 따라 `extra` 를 정식 필드로 승격한다.

## 2. Modbus TCP 매핑 초안 (기종 확정 후 수정)

| 레지스터 | 방향 | 내용 |
|---|---|---|
| Coil 0..N-1 | PC→PLC | 계란 위치별 배출 플래그 (판 단위로 한 번에 write) |
| HR 100 | PC→PLC | 현재 판 시퀀스 번호 |
| HR 101 | PC→PLC | 하트비트 카운터 (1초마다 증가) |
| HR 102 | PC→PLC | 알람 코드 (0=정상) |
| Coil 200 | PC→PLC | 라인 정지 요청 |
| HR 200 | PLC→PC | 판 도착 트리거 / 인코더 카운트 |

## 3. Mock 동작

- 모든 호출을 `calls` 리스트에 `(이름, 페이로드)` 로 기록
- `fail_connect=True`: 연결 실패 시뮬레이션
- `fail_after_calls=n`: n 번째 명령 이후 링크 끊김 시뮬레이션 → 파이프라인 안전 동작 테스트에 사용

## 4. 통합 테스트 계획

1. 실제 PLC 시뮬레이터(예: pymodbus 서버)에 대해 같은 테스트 케이스 실행
2. 하트비트 중단 시 PLC 측 안전 동작이 발동하는지 현장 확인
3. 배출 타이밍 오차 측정: 카메라 트리거 → 배출 작동까지 지연 분포
