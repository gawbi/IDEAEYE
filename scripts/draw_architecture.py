# -*- coding: utf-8 -*-
"""소프트웨어 아키텍처 다이어그램 v2 — 제품 로고 블록 + 프로토콜 화살표 + 하단 워크플로 스트립.

python scripts/draw_architecture.py                      # 검수/10_웹모니터링/architecture.png, docs/web_architecture.png 갱신
python scripts/draw_architecture.py out1.png out2.png    # 다른 경로로 저장
아이콘: scripts/assets/icons/*.png (simpleicons.org, CC0). 폰트: Malgun Gothic.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Wedge

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON = os.path.join(ROOT, "scripts", "assets", "icons")

W, H = 220, 176
fig = plt.figure(figsize=(22, 17.6), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

DONE, PREP, PLAN = "#223a5e", "#d97b1e", "#8c8c8c"  # 구현 / 실장 대기 / 계획
SRV, EDGE, DEV = "#3b6ea5", "#4e8f3a", "#7a5aa6"
UP, DOWN = "#b03a2e", "#1e8449"
TXT = "#2a2a2a"
PX = fig.get_figwidth() * fig.dpi / W  # 1 data unit = PX pixels (at fig dpi)


def icon(name, x, y, size=4.6):
    p = os.path.join(ICON, f"{name}.png")
    if not os.path.exists(p):
        return
    img = mpimg.imread(p)
    zoom = size * PX / img.shape[0]
    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y), frameon=False))


def box(x, y, w, h, title, lines=(), icons=(), status=DONE, fs=9.3, tfs=12, fc="white", lh=3.8):
    dashed = status == PLAN
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8", fc=fc, ec=status,
                                lw=1.6 if dashed else 2.1, ls=(0, (5, 3)) if dashed else "-", zorder=2))
    ix = x + 3.2
    for nm in icons:
        icon(nm, ix + 2.3, y + h - 4.0, 4.4)
        ix += 5.6
    tx = ix + 0.6 if icons else x + 3.2
    ax.text(tx, y + h - 4.0, title, ha="left", va="center", fontsize=tfs, fontweight="bold", color="#111", zorder=3)
    for i, t in enumerate(lines):
        ax.text(x + 3.2, y + h - 8.6 - i * lh, t, ha="left", va="center", fontsize=fs, color=TXT, zorder=3)


def group(x, y, w, h, title, color, icons=(), sub=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3", fc=color + "0d", ec=color, lw=1.6, ls=(0, (7, 4)), zorder=1))
    ix = x + 4
    for nm in icons:
        icon(nm, ix + 2.4, y + h - 3.6, 4.4)
        ix += 5.8
    ax.text(ix + 0.8 if icons else x + 4, y + h - 3.6, title, ha="left", va="center", fontsize=13.5, fontweight="bold", color=color, zorder=3)
    if sub:
        ax.text(x + w - 4, y + h - 3.6, sub, ha="right", va="center", fontsize=9.5, color=color, zorder=3)


def label(x, y, text, color, fs=8.8, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, lw=0.7))


def arrow(p1, p2, text="", color="#333", style="-|>", ls="-", tpos=0.5, toff=(0, 2.0), fs=8.8, lw=1.7):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=17, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))
    if text:
        label(p1[0] + (p2[0] - p1[0]) * tpos + toff[0], p1[1] + (p2[1] - p1[1]) * tpos + toff[1], text, color, fs)


def polyarrow(pts, color="#333", ls="-", lw=1.7, head=True):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=4, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=17, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))


# ───────────────────────── 제목 · 범례
ax.text(6, 170.5, "AI 파각 검출 · 검란 시스템 — 소프트웨어 아키텍처", fontsize=21, fontweight="bold", color="#111", va="center")
ax.text(6, 165.4, "파각란검출기(Jetson Orin Nano) → 무선(WiFi / LTE Cat.M1) → 모니터링 서버(FastAPI) → 대시보드 · KS B 7956-4 송출     |     학습 파이프라인 → Git → 배포",
        fontsize=10.5, color="#555", va="center")
lx = 150
for i, (c, t, ls) in enumerate([(DONE, "구현 · 단위 테스트 완료", "-"), (PREP, "mock 구현 · 기기 입고 후 실장", "-"), (PLAN, "계획 (미구현)", (0, (5, 3)))]):
    yy = 172.2 - i * 3.3
    ax.plot([lx, lx + 7], [yy, yy], color=c, lw=2.2, ls=ls)
    ax.text(lx + 9, yy, t, fontsize=9.5, color="#333", va="center")
ax.text(214, 162.6, "2026-10-09 기준 · 상세: docs/web_monitoring_design.md · docs/phase7_deployment.md · docs/field_info.md", fontsize=8.2, color="#777", ha="right", va="center")

# ───────────────────────── 상단: 개발 단계 (데이터 → 학습 → Git)
TY, TH = 141, 19
box(6, TY, 52, TH, "데이터 수집 · 라벨링", [
    "AI Hub 계란 데이터 94만 크롭 (MONO/COLOR)",
    "기기 영상 → 크롭  scripts/ingest_video.py",
    "라벨링 도구 label_crops.py · 통합 own_build_index.py"], icons=["opencv", "numpy"], status=DONE)
box(64, TY, 58, TH, "학습 파이프라인 (개발 PC · RTX 4060)", [
    "YOLO11n 계란 검출 (Ultralytics) · EfficientNet 크랙/소분류/심각도 (timm)",
    "그룹 분할(촬영일·지역) · 운영점 평가(FPR@Recall95) · experiments/ 기록",
    "ONNX 내보내기 (fixed batch 6) → TensorRT 변환은 Jetson 에서"], icons=["pytorch", "ultralytics", "onnx"], status=DONE)
box(128, TY, 52, TH, "GitHub Repository (비공개 예정)", [
    "현재 로컬 git 커밋 · 원격 저장소 계정 필요",
    "src/ configs/ deploy/ tests/ docs/ 검수/",
    "CI: pytest 자동 실행 · Docker 이미지 빌드 (계획)"], icons=["github", "docker"], status=PLAN)
arrow((58, TY + 9.5), (64, TY + 9.5), "크롭 · index.csv", DEV, toff=(0, 2.6), fs=8.2)
arrow((122, TY + 9.5), (128, TY + 9.5), "코드 · 설정 · 모델", DEV, toff=(0, 2.6), fs=8.2)

# Git → 서버 / 검출기 배포
arrow((154, TY), (154, 136), "git pull · Docker 배포", DEV, toff=(14, 0))
polyarrow([(180, TY + 9.5), (215, TY + 9.5), (215, 46), (212, 46)], color=DEV)
ax.text(217.6, 98, "deploy/jetson: setup.sh · export_onnx.py · build_trt.sh → 모델(.engine) · 코드 배포", rotation=90, fontsize=8.5, color=DEV, ha="center", va="center")

# ───────────────────────── 모니터링 서버 그룹
group(46, 82, 166, 54, "모니터링 서버 (학교 / 클라우드)", SRV, icons=["python", "docker"], sub="Docker 1명령 배포 · PostgreSQL 전환 가능 (계획)")
box(50, 105, 42, 24, "Front-end (대시보드)", [
    "/  장비 목록 · 온라인 상태 · 오늘 실적",
    "/devices/{id}  일·시간 추이 · 심각도 · 이벤트",
    "Jinja2 템플릿 + Chart.js (React 전환 가능)",
    "/docs  API 문서 (OpenAPI)"], icons=["jinja", "chartdotjs", "html5"], status=DONE)
box(100, 105, 56, 24, "API Server (FastAPI · Uvicorn)", [
    "장비 API  POST heartbeat · counts(upsert) · events · eggs (멱등)",
    "설정 API  GET config (버전 · 설정 시각)   관리 API  장비 등록(api_key) · 설정 변경",
    "조회 API  devices · summary   KS 송출  GET /api/v1/export/ks → CSV",
    "인증  장비 Bearer api_key(해시 저장) · 관리자 X-Admin-Token · 로그인/HTTPS(계획)"], icons=["fastapi", "python"], status=DONE, fs=8.8)
box(164, 105, 44, 24, "Database (SQLite)", [
    "devices · heartbeats · daily_counts",
    "hourly_counts · events · egg_logs",
    "SQLAlchemy ORM · 파일 1개 (백업 용이)",
    "→ PostgreSQL 교체 가능"], icons=["sqlite", "sqlalchemy"], status=DONE)
arrow((92, 119), (100, 119), "", SRV, style="<|-|>")
label(96, 123.5, "HTTP / JSON\n(REST · Jinja2 렌더링)", SRV, fs=8.2)
arrow((156, 119), (164, 119), "", SRV, style="<|-|>")
label(160, 123.5, "ORM\n(SQLAlchemy)", SRV, fs=8.2)

box(50, 86, 42, 15, "알림 서비스 (계획)", [
    "알람 발생 · 오프라인 N분 → 카카오/문자/이메일",
    "events 테이블 기반 · 임계 파각률 초과 알림"], status=PLAN, fs=8.8)
box(164, 86, 44, 15, "KS B 7956-4 송출 데이터", [
    "검출기 ID · 처리란/파각란 개수(일) · 설정 시각",
    "daily_counts + config_set_ts → CSV (외부 보고)"], status=DONE, fs=8.8)
arrow((71, 105), (71, 101), "", PLAN, ls=(0, (4, 3)))
label(71, 103, "이벤트 트리거", PLAN, fs=7.8)
arrow((186, 105), (186, 101), "", SRV)
label(186, 103, "일별 집계 조회", SRV, fs=7.8)

# 관리자 (브라우저)
ax.add_patch(Circle((23, 124), 3.2, fc="#e8eef6", ec=DONE, lw=1.8, zorder=3))
ax.add_patch(Wedge((23, 112.5), 6.8, 0, 180, fc="#e8eef6", ec=DONE, lw=1.8, zorder=3))
ax.text(23, 108.5, "관리자 / 운영자", ha="center", va="center", fontsize=11, fontweight="bold", color="#111")
ax.text(23, 105.2, "브라우저 (PC · 모바일)", ha="center", va="center", fontsize=9, color="#555")
ax.text(23, 102.3, "장비 상태 확인 · 임계값 변경 · CSV 내보내기", ha="center", va="center", fontsize=8.2, color="#555")
arrow((31.5, 117), (50, 117), "", SRV, style="<|-|>")
label(39, 120, "HTTPS", SRV, fs=8.2)

# ───────────────────────── 무선 구간 (서버 ↔ 검출기)
polyarrow([(187, 66), (187, 78), (116, 78), (116, 105)], color=UP)        # 업링크 (텔레메트리 → API)
polyarrow([(140, 105), (140, 75), (197, 75), (197, 66)], color=DOWN)      # 다운링크 (API → 텔레메트리)
label(112, 93, "▲ 업링크 HTTPS (Bearer api_key)\nPOST heartbeat · counts · events · eggs\n집계 ~1 KB/일 · 멱등 재전송 · 백오프", UP, fs=8.2)
label(144, 93, "▼ 다운링크 GET config 폴링\n임계값 · 투표 규칙 · 설정 버전\n설정 시각(YYYYMMDDhhmmss)", DOWN, fs=8.2)
label(80, 76.5, "인터넷  WiFi (기본) / LTE Cat.M1 (옵션) · HTTPS 기본 · MQTT 어댑터 가능", "#555", fs=9.2)

# ───────────────────────── 파각란검출기 그룹
group(46, 15, 166, 57, "파각란검출기 (현장 · Jetson Orin Nano · JetPack / Ubuntu)", EDGE, icons=["nvidia", "linux"],
      sub="")
ax.text(178, 68.4, "계란당 추론 ~2 ms (RTX 4060 기준 · Jetson 실측 예정)", ha="right", va="center", fontsize=9.5, color=EDGE, zorder=3)
RA, RAH = 38, 28   # 윗줄
RB, RBH = 19, 15   # 아랫줄
box(50, RA, 40, RAH, "카메라 입력 (캡처 모듈 계획)", [
    "ELP IMX678 ×2 · 1/1.8\" · 화각 94°",
    "USB 3.0 UVC · 4K60 MJPEG",
    "cv2.VideoCapture / GStreamer",
    "하드웨어 디코딩 · 2대 동기화",
    "회전 신호 → 3시점 프레임 선택",
    "노출 · flat-field 보정 (옵션)"], icons=["opencv"], status=PLAN, fs=8.8, lh=3.3)
box(102, RA, 56, RAH, "추론 파이프라인  src/infer", [
    "① 고정 ROI 크롭 (또는 YOLO11n 계란 위치 검출)",
    "② 크랙 vs 정상 2클래스 헤드 · EfficientNet-B0 · TensorRT FP16 · batch 6",
    "③ 소분류 · 심각도 헤드 (파각 정도 1~3등급, 물리 지표 결합 예정)",
    "④ 3시점 투표 EggAggregator (max / hybrid 규칙 · 타임아웃 처리)",
    "⑤ 최종 판정 → 배출 신호 (PLC) · 판정 로그 JSONL (계란 ID · 3시점 점수)",
    "⑥ 안전 동작: 카메라 · 모델 · PLC 장애 시 배출 정지 + 이벤트"],
    icons=["pytorch", "onnx", "nvidia"], status=DONE, fs=8.6, lh=3.3)
box(166, RA, 42, RAH, "텔레메트리  src/telemetry", [
    "Aggregator  일 · 시간별 처리란/파각란 집계",
    "LocalQueue  SQLite 큐 (kind, key) upsert",
    "    오프라인 시 보관 → 복구 후 재전송",
    "Uploader  HTTPS 전송 · 지수 백오프",
    "    설정 폴링 (임계값 · 투표 규칙 · 버전)",
    "KS 송출 항목 집계 (검출기 ID · 일별 개수)"], icons=["python"], status=DONE, fs=8.6, lh=3.3)
box(50, RB, 40, RBH, "로컬 저장", [
    "판정 로그 JSONL · 오판 의심 크롭 (재학습용)",
    "전송 큐 SQLite (텔레메트리 LocalQueue)"], icons=["sqlite"], status=DONE, fs=8.6)
box(102, RB, 56, RBH, "회전 · 배출 제어 (PLC)", [
    "입력  회전 신호 · 사이클 번호     출력  배출 신호 (파각 판정 시)",
    "PLC 기종 · 통신(GPIO / RS-485) 미확정 → src/plc mock (실장 대기)"], status=PREP, fs=8.6)
box(166, RB, 42, RBH, "상태 감시 · 하트비트", [
    "온라인 · 모델/설정 버전 · 장애 상태 주기 전송",
    "카메라/모델/PLC 장애 이벤트 → 서버 알림"], status=DONE, fs=8.6)
arrow((90, RA + 14), (102, RA + 14), "", EDGE)
label(96, RA + 18, "USB 3.0\n프레임", EDGE, fs=7.8)
arrow((158, RA + 14), (166, RA + 14), "", EDGE)
label(162, RA + 18.3, "판정 결과\n(계란 단위)", EDGE, fs=7.8)
arrow((130, RA), (130, RB + RBH), "", PREP)
label(130, RA - 2, "배출 신호", PREP, fs=7.8)
arrow((187, RB + RBH), (187, RA), "", EDGE)
label(187, RA - 2, "하트비트 · 이벤트", EDGE, fs=7.8)
arrow((102, RA + 3), (90, RB + RBH - 1.5), "", EDGE, lw=1.4)
label(96, RA - 2.2, "판정 로그", EDGE, fs=7.8)

# 계란 (현장 입력)
for r, col in enumerate(["#c9a27a", "#f6f1e8"]):
    for c in range(3):
        ax.add_patch(Ellipse((15 + c * 7.5, 58 - r * 9), 5.4, 7.0, fc=col, ec="#7a6a55", lw=1.2, zorder=3))
ax.text(22.5, 43.5, "계란 (황색란 · 흰색란)", ha="center", va="center", fontsize=11, fontweight="bold", color="#111")
ax.text(22.5, 40.2, "투광 촬영 · 자리 3개 × 카메라 2대", ha="center", va="center", fontsize=8.6, color="#555")
ax.text(22.5, 37.3, "120° 회전 × 3회 촬영 후 최종 판정", ha="center", va="center", fontsize=8.6, color="#555")
arrow((33, 53.5), (50, 53.5), "", EDGE)
label(41.5, 56.5, "선별 라인", EDGE, fs=8.2)

# 재학습 루프 (오판 크롭 → 라벨링)
polyarrow([(50, RB + 4), (43.5, RB + 4), (43.5, TY)], color=PLAN, ls=(0, (4, 3)), lw=1.4)
ax.text(41.6, 86, "오판 의심 크롭 → 재라벨링 → 미세조정 (재학습 루프, 계획)", rotation=90, fontsize=8.2, color=PLAN, ha="center", va="center")

# ───────────────────────── 하단 워크플로 스트립
ax.add_patch(FancyBboxPatch((4, 0.6), 212, 12.4, boxstyle="round,pad=0,rounding_size=2", fc="#f2f4f7", ec="#cfd4dc", lw=1.2, zorder=1))
steps = [
    ("촬영", "자리 3개 × 120° 회전 3회\n카메라 2대 · USB 3.0"),
    ("ROI · 크롭", "고정 ROI 또는 YOLO11n\n계란 위치 검출 · 320px"),
    ("추론", "크랙 vs 정상 + 심각도\nTensorRT FP16 · batch 6"),
    ("3시점 투표", "max / hybrid 규칙\n최종 판정 (목표 Recall ≥ 95%)"),
    ("배출", "PLC 배출 신호 · 안전 동작\n판정 로그 JSONL"),
    ("집계 · 전송", "일·시간별 집계 → HTTPS\nWiFi / LTE · 오프라인 큐"),
    ("모니터링", "대시보드 · 알림\nKS B 7956-4 송출 CSV"),
]
x0, dx = 18, 30.5
for i, (t, d) in enumerate(steps):
    cx = x0 + i * dx
    ax.add_patch(Circle((cx - 9, 9.6), 2.3, fc=DONE, ec=DONE, zorder=3))
    ax.text(cx - 9, 9.6, str(i + 1), ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=4)
    ax.text(cx - 5.6, 9.6, t, ha="left", va="center", fontsize=11, fontweight="bold", color="#111", zorder=4)
    ax.text(cx - 11.5, 4.3, d, ha="left", va="center", fontsize=8.2, color="#444", zorder=4)
    if i < len(steps) - 1:
        ax.annotate("", (cx + dx - 12.5, 9.6), (cx + 15.5, 9.6), arrowprops=dict(arrowstyle="-|>", color="#9aa3b2", lw=1.4, mutation_scale=14), zorder=3)

outs = sys.argv[1:] or [os.path.join(ROOT, "검수", "10_웹모니터링", "architecture.png"), os.path.join(ROOT, "docs", "web_architecture.png")]
for out in outs:
    fig.savefig(out, dpi=140, facecolor="white")
    print("saved", out)
