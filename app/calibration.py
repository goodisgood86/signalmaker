from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import json
import math


@dataclass(frozen=True)
class PlattModel:
    a: float
    b: float


@dataclass(frozen=True)
class IsotonicModel:
    xs: List[float]
    ys: List[float]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_platt(
    p_raw: Iterable[float],
    y: Iterable[int],
    *,
    epochs: int = 2000,
    lr: float = 0.05,
) -> PlattModel:
    xs = [min(1.0 - 1e-6, max(1e-6, float(v))) for v in p_raw]
    ys = [1.0 if int(v) == 1 else 0.0 for v in y]
    n = len(xs)
    if n == 0 or n != len(ys):
        return PlattModel(a=1.0, b=0.0)

    a = 1.0
    b = 0.0

    for _ in range(epochs):
        da = 0.0
        db = 0.0
        for x, t in zip(xs, ys):
            p = _sigmoid(a * x + b)
            e = p - t
            da += e * x
            db += e
        da /= n
        db /= n
        a -= lr * da
        b -= lr * db

    return PlattModel(a=float(a), b=float(b))


def apply_platt(model: PlattModel, p_raw: float) -> float:
    x = min(1.0 - 1e-6, max(1e-6, float(p_raw)))
    return min(1.0, max(0.0, _sigmoid(model.a * x + model.b)))


def save_platt_model(model: PlattModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "platt",
        "a": model.a,
        "b": model.b,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_platt_model(path: str | Path) -> PlattModel | None:
    p = Path(path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("type") != "platt":
        return None
    a = float(raw.get("a", 1.0))
    b = float(raw.get("b", 0.0))
    return PlattModel(a=a, b=b)


def fit_isotonic(p_raw: Iterable[float], y: Iterable[int]) -> IsotonicModel:
    pairs = sorted(
        (min(1.0 - 1e-6, max(1e-6, float(p))), float(1 if int(t) == 1 else 0))
        for p, t in zip(p_raw, y)
    )
    if not pairs:
        return IsotonicModel(xs=[0.0, 1.0], ys=[0.5, 0.5])

    # Pool Adjacent Violators (PAV): 단조 증가 제약
    blocks = [{"sum_w": 1.0, "sum_y": yv, "left": xv, "right": xv} for xv, yv in pairs]
    i = 0
    while i < len(blocks) - 1:
        mean_i = blocks[i]["sum_y"] / blocks[i]["sum_w"]
        mean_j = blocks[i + 1]["sum_y"] / blocks[i + 1]["sum_w"]
        if mean_i <= mean_j:
            i += 1
            continue
        blocks[i]["sum_w"] += blocks[i + 1]["sum_w"]
        blocks[i]["sum_y"] += blocks[i + 1]["sum_y"]
        blocks[i]["right"] = blocks[i + 1]["right"]
        del blocks[i + 1]
        if i > 0:
            i -= 1

    xs: List[float] = []
    ys_out: List[float] = []
    for b in blocks:
        left = float(b["left"])
        right = float(b["right"])
        val = float(b["sum_y"] / b["sum_w"])
        xs.extend([left, right])
        ys_out.extend([val, val])

    if len(xs) == 1:
        xs = [0.0, 1.0]
        ys_out = [ys_out[0], ys_out[0]]

    return IsotonicModel(xs=xs, ys=ys_out)


def apply_isotonic(model: IsotonicModel, p_raw: float) -> float:
    x = min(1.0 - 1e-6, max(1e-6, float(p_raw)))
    xs = model.xs
    ys = model.ys
    if not xs or not ys:
        return x
    if x <= xs[0]:
        return max(0.0, min(1.0, ys[0]))
    if x >= xs[-1]:
        return max(0.0, min(1.0, ys[-1]))
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 <= x0:
                return max(0.0, min(1.0, y1))
            t = (x - x0) / (x1 - x0)
            y = y0 + t * (y1 - y0)
            return max(0.0, min(1.0, y))
    return max(0.0, min(1.0, ys[-1]))


def save_isotonic_model(model: IsotonicModel, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "isotonic",
        "xs": model.xs,
        "ys": model.ys,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_isotonic_model(path: str | Path) -> IsotonicModel | None:
    p = Path(path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("type") != "isotonic":
        return None
    xs = [float(v) for v in raw.get("xs", [])]
    ys = [float(v) for v in raw.get("ys", [])]
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return IsotonicModel(xs=xs, ys=ys)
