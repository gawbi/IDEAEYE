"""Phase 3-B / 4 / 5: 계란 크롭 분류기 (timm). 소분류 코드('state/state_detail') 단위 학습·평가.

설정은 configs/exp_cls_*.yaml. 전처리(원본/CLAHE/언샤프)·손실(CE/가중CE/focal)·해상도는 ablation 용 옵션.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..eval.metrics import operating_point, per_class_metrics

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- 전처리


def preprocess_image(img: np.ndarray, mode: str) -> np.ndarray:
    """mode: none | clahe | unsharp | clahe+unsharp. BGR uint8 → BGR uint8."""
    if mode in ("clahe", "clahe+unsharp"):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if mode in ("unsharp", "clahe+unsharp"):
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s != 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = img
    return out


def imread(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], img_size: int, train: bool, preprocess: str = "none", label_col: str = "cls"):
        self.paths = df["crop_path"].tolist()
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[c] for c in df[label_col].astype(str)]
        self.img_size = img_size
        self.train = train
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = imread(self.paths[i])
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if self.preprocess != "none":
            img = preprocess_image(img, self.preprocess)
        img = letterbox(img, self.img_size)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
            if np.random.rand() < 0.8:
                img = np.clip(img.astype(np.float32) * np.random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
        x = (img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), self.labels[i]


# ---------------------------------------------------------------- 손실


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        w = self.weight[target] if self.weight is not None else 1.0
        pt = p.gather(1, target[:, None]).squeeze(1)
        loss = -w * (1 - pt) ** self.gamma * logp.gather(1, target[:, None]).squeeze(1)
        return loss.mean()


def make_loss(kind: str, class_counts: list[int], device: torch.device, label_smoothing: float = 0.0) -> nn.Module:
    counts = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1)
    if kind == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if kind == "weighted_ce":  # sqrt 역빈도 가중 (과도한 가중 방지)
        w = (counts.max() / counts).sqrt()
        return nn.CrossEntropyLoss(weight=(w / w.mean()).to(device), label_smoothing=label_smoothing)
    if kind == "focal":
        return FocalLoss(gamma=2.0)
    if kind == "weighted_focal":
        w = (counts.max() / counts).sqrt()
        return FocalLoss(gamma=2.0, weight=(w / w.mean()).to(device))
    raise ValueError(kind)


# ---------------------------------------------------------------- 데이터 준비


def load_index(index_csv: Path, light: str, label_col: str = "cls") -> pd.DataFrame:
    df = pd.read_csv(index_csv, dtype={"state": str, "state_detail": str, "cls": str, "washed": str, "breed": str, "region_code": str}, encoding="utf-8-sig")
    df = df[(df["light"] == light) & (df["exists"] == True)]  # noqa: E712
    return df[df["split"].notna()].copy()


def cap_per_class(df: pd.DataFrame, cap: int | None, seed: int, label_col: str = "cls") -> pd.DataFrame:
    if not cap:
        return df
    parts = [g.sample(n=min(cap, len(g)), random_state=seed) for _, g in df.groupby(label_col)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


# ---------------------------------------------------------------- 학습/평가


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        probs.append(F.softmax(out.float(), dim=1).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(ys)


def evaluate(probs: np.ndarray, y: np.ndarray, classes: list[str], meta: pd.DataFrame | None = None, group_cols: list[str] | None = None) -> dict[str, Any]:
    pred = probs.argmax(1)
    y_names = [classes[i] for i in y]
    p_names = [classes[i] for i in pred]
    out: dict[str, Any] = {"overall": per_class_metrics(y_names, p_names, classes), "accuracy": float((pred == y).mean())}
    out["macro_recall"] = float(np.mean([m["recall"] for m in out["overall"].values() if m["support"] > 0]))
    if meta is not None and group_cols:
        out["by"] = {}
        for col in group_cols:
            if col in meta:
                out["by"][col] = {}
                for val, idx in meta.groupby(col).indices.items():
                    out["by"][col][str(val)] = per_class_metrics([y_names[i] for i in idx], [p_names[i] for i in idx], classes)
    return out


def crack_operating_points(probs: np.ndarray, y: np.ndarray, classes: list[str], crack_prefix: str = "2/", normal_prefix: str = "1/", targets=(0.90, 0.95, 0.98)) -> dict[str, Any]:
    """파각 점수 = 크랙 소분류 확률 합. 양성=크랙, 음성=정상 → Recall 목표별 정상란 오검출률."""
    def _is(c: str, prefix: str) -> bool:  # "2/6"(소분류) 또는 "2"(대분류 코드) 모두 허용
        return c.startswith(prefix) or c == prefix.rstrip("/")

    crack_idx = [i for i, c in enumerate(classes) if _is(c, crack_prefix)]
    normal_idx = [i for i, c in enumerate(classes) if _is(c, normal_prefix)]
    if not crack_idx or not normal_idx:
        return {}
    score = probs[:, crack_idx].sum(1)
    is_crack = np.isin(y, crack_idx)
    is_normal = np.isin(y, normal_idx)
    keep = is_crack | is_normal
    out = {}
    for t in targets:
        op = operating_point(score[keep], is_crack[keep], t)
        # 같은 임계값에서 소분류별 Recall
        thr = op["threshold"]
        sub = {}
        for i in crack_idx:
            m = y == i
            if m.any():
                sub[classes[i]] = float((score[m] >= thr).mean())
        op["subclass_recall"] = sub
        out[f"recall@{t:.2f}"] = op
    return out


def train_classifier(cfg: dict[str, Any], index_csv: Path, out_dir: Path, device: torch.device | None = None, log=print) -> dict[str, Any]:
    import timm

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    light, label_col = cfg["light"], cfg.get("label_col", "cls")
    df = load_index(index_csv, light, label_col)
    if cfg.get("class_filter"):  # 예: 혈반 소분류만 ("3/") — 필터는 항상 cls(소분류 코드) 기준
        df = df[df["cls"].str.startswith(tuple(cfg["class_filter"]))]
    if cfg.get("label_map"):  # 코드 → 새 라벨 (기본 state 기준; label_map_col 로 state_detail/cls 등 지정 가능. "drop" 값은 제외)
        lm = {str(k): str(v) for k, v in cfg["label_map"].items()}
        src_col = cfg.get("label_map_col", "state")
        df = df.copy()
        df[label_col] = df[src_col].astype(str).map(lambda s: lm.get(s, lm.get("default", s)))
        df = df[df[label_col] != "drop"]
    df[label_col] = df[label_col].astype(str)

    def _key(v: str):
        try:
            return (0, tuple(int(x) for x in v.split("/")))
        except ValueError:
            return (1, (v,))

    classes = sorted(df[label_col].unique(), key=_key)
    train_df = cap_per_class(df[df["split"] == "train"], cfg.get("train_cap_per_class"), cfg["seed"], label_col)
    val_df = df[df["split"] == "val"]
    if cfg.get("val_max") and len(val_df) > cfg["val_max"]:
        val_df = val_df.sample(n=cfg["val_max"], random_state=cfg["seed"])
    test_df = df[df["split"] == "test"]
    if cfg.get("test_max") and len(test_df) > cfg["test_max"]:
        test_df = test_df.sample(n=cfg["test_max"], random_state=cfg["seed"])
    counts = Counter(train_df[label_col])
    log(f"[{light}] classes={len(classes)} train={len(train_df)} val={len(val_df)} test={len(test_df)} device={device}")

    mk = lambda d, train: DataLoader(CropDataset(d, classes, cfg["img_size"], train, cfg.get("preprocess", "none"), label_col), batch_size=cfg["batch"], shuffle=train, num_workers=cfg.get("workers", 8), pin_memory=True, persistent_workers=cfg.get("workers", 8) > 0, drop_last=train)
    train_loader, val_loader, test_loader = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    model = timm.create_model(cfg["model"], pretrained=True, num_classes=len(classes)).to(device)
    if cfg.get("init_from"):  # 기존 체크포인트에서 시작 (Phase 6 미세조정)
        ck = torch.load(cfg["init_from"], map_location=device, weights_only=False)
        if ck["classes"] == classes:
            model.load_state_dict(ck["state_dict"])
        else:  # 클래스 집합이 다르면 백본만
            sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith(("classifier", "head", "fc"))}
            model.load_state_dict(sd, strict=False)
    loss_fn = make_loss(cfg.get("loss", "ce"), [counts.get(c, 0) for c in classes], device, cfg.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    steps = cfg["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(1, steps), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, best_state, history = -1.0, None, []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), torch.as_tensor(y).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        probs, ys = predict(model, val_loader, device)
        ev = evaluate(probs, ys, classes)
        history.append({"epoch": epoch + 1, "train_loss": tot / max(1, n), "val_acc": ev["accuracy"], "val_macro_recall": ev["macro_recall"], "sec": time.time() - t0})
        log(f"  epoch {epoch + 1}/{cfg['epochs']} loss={tot / max(1, n):.4f} val_acc={ev['accuracy']:.4f} val_macro_recall={ev['macro_recall']:.4f} ({time.time() - t0:.0f}s)")
        if ev["macro_recall"] > best:
            best, best_state = ev["macro_recall"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:  # epochs=0 + init_from: 학습 없이 체크포인트 평가만
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "cfg": cfg}, out_dir / "model.pt")

    probs, ys = predict(model, test_loader, device)
    meta = test_df.reset_index(drop=True)
    ev = evaluate(probs, ys, classes, meta, cfg.get("group_cols", ["washed", "breed", "region_code"]))
    ev["operating_points"] = crack_operating_points(probs, ys, classes)
    # 3클래스 등 '기타 결함' 라벨이 있으면: 같은 운영점에서 기타 결함이 크랙으로 판정되는 비율도 기록 (배출 정책 시뮬레이션용)
    other_idx = [i for i, c in enumerate(classes) if not (c.startswith(("1/", "2/")) or c in ("1", "2"))]
    if other_idx and ev["operating_points"]:
        crack_idx = [i for i, c in enumerate(classes) if c.startswith("2/") or c == "2"]
        score = probs[:, crack_idx].sum(1)
        is_other = np.isin(ys, other_idx)
        for k, op in ev["operating_points"].items():
            op["other_flagged_as_crack"] = float((score[is_other] >= op["threshold"]).mean()) if is_other.any() else None
    ev["history"] = history
    ev["classes"] = classes
    ev["n"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    # 예측 저장 (FN 분석·운영점 재계산용)
    pred_df = meta[["crop_path", label_col, "tray_id", "seq"] + [c for c in ["washed", "breed", "region_code", "date"] if c in meta]].copy()
    pred_df["y_pred"] = [classes[i] for i in probs.argmax(1)]
    pred_df["p_max"] = probs.max(1)
    crack_idx = [i for i, c in enumerate(classes) if c.startswith("2/") or c == "2"]
    if crack_idx:
        pred_df["p_crack"] = probs[:, crack_idx].sum(1)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "test_probs.npy", probs.astype(np.float16))
    (out_dir / "eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    return ev


@torch.no_grad()
def benchmark_latency(model: nn.Module, img_size: int, device: torch.device, batch: int = 1, iters: int = 50) -> dict[str, float]:
    model.eval()
    x = torch.randn(batch, 3, img_size, img_size, device=device, dtype=next(model.parameters()).dtype)
    for _ in range(5):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return {"batch": batch, "ms_per_batch": ms, "ms_per_image": ms / batch}
