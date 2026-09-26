"""실제 장비 없이 파이프라인을 테스트하기 위한 PLC mock. 모든 호출을 기록하고 장애를 시뮬레이션한다."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import AlarmCode, PLCError, PLCInterface, RejectCommand


@dataclass
class MockPLC(PLCInterface):
    fail_connect: bool = False
    fail_after_calls: int | None = None  # n 번째 명령부터 PLCError 발생 (연결 끊김 시뮬레이션)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    _connected: bool = False
    _n_cmds: int = 0

    def _record(self, name: str, payload: Any = None) -> None:
        self.calls.append((name, payload))

    def _guard(self) -> None:
        if not self._connected:
            raise PLCError("not connected")
        self._n_cmds += 1
        if self.fail_after_calls is not None and self._n_cmds > self.fail_after_calls:
            self._connected = False
            raise PLCError("simulated link loss")

    def connect(self) -> None:
        self._record("connect")
        if self.fail_connect:
            raise PLCError("simulated connect failure")
        self._connected = True

    def disconnect(self) -> None:
        self._record("disconnect")
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_reject(self, cmd: RejectCommand) -> None:
        self._guard()
        self._record("send_reject", cmd)

    def reject_all(self, tray_seq: int, reason: str) -> None:
        self._guard()
        self._record("reject_all", {"tray_seq": tray_seq, "reason": reason})

    def stop_line(self, reason: str) -> None:
        self._guard()
        self._record("stop_line", reason)

    def heartbeat(self) -> None:
        self._guard()
        self._record("heartbeat")

    def alarm(self, code: AlarmCode, message: str) -> None:
        # 알람은 연결 상태와 무관하게 기록 (로컬 경보·로그 대체)
        self._record("alarm", {"code": code.value, "message": message})

    def reset_alarm(self) -> None:
        self._record("reset_alarm")

    # 테스트 편의
    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def rejects(self) -> list[RejectCommand]:
        return [p for n, p in self.calls if n == "send_reject"]
