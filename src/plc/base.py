"""PLC 배출 연동 인터페이스. 실제 구현(Modbus TCP 등)과 MockPLC 가 같은 계약을 따른다.

설계 문서: docs/plc_interface.md
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PLCError(RuntimeError):
    """통신 실패·타임아웃 등. 파이프라인은 이 예외를 잡아 safety 정책을 적용한다."""


class AlarmCode(str, Enum):
    MODEL_LOAD_FAIL = "E100"
    CAMERA_LOST = "E200"
    PLC_LOST = "E300"
    LATENCY_EXCEEDED = "W400"
    INFERENCE_ERROR = "E500"


@dataclass
class RejectCommand:
    """판 내 계란 위치(행·열 또는 인덱스)와 사유. 실제 배출 기구에 맞게 lane/timing 필드를 추가한다."""

    tray_seq: int
    egg_index: int
    reason: str
    score: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class PLCInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def send_reject(self, cmd: RejectCommand) -> None:
        """단일 계란 배출 신호."""

    @abstractmethod
    def reject_all(self, tray_seq: int, reason: str) -> None:
        """판 전량 배출 (안전 동작)."""

    @abstractmethod
    def stop_line(self, reason: str) -> None:
        """라인 정지 요청 (안전 동작)."""

    @abstractmethod
    def heartbeat(self) -> None:
        """주기적 생존 신호. 끊기면 PLC 측이 안전 동작을 수행하도록 설계한다."""

    @abstractmethod
    def alarm(self, code: AlarmCode, message: str) -> None: ...

    @abstractmethod
    def reset_alarm(self) -> None: ...
