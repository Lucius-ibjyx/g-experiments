from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def read_ergo_csv(path: str | Path) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Read ErgoLAB-export CSVs that begin with a short metadata block.

    The actual CSV header is detected by the line beginning with `Id,Time(s)`.
    Returns (metadata_dict, dataframe).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header_idx = None
    meta: Dict[str, str] = {}
    for i, line in enumerate(text):
        if line.startswith("Id,Time(s)"):
            header_idx = i
            break
        if "," in line:
            k, v = line.split(",", 1)
            meta[k.strip()] = v.strip()
    if header_idx is None:
        raise ValueError(f"Cannot locate data header in {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=header_idx)
    return meta, df


def probe_video(path: str | Path) -> dict:
    path = Path(path)
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    s = data["streams"][0]

    def ratio(v: str | None):
        if not v:
            return None
        if "/" in v:
            a, b = v.split("/")
            return float(a) / float(b)
        return float(v)

    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "avg_fps": ratio(s.get("avg_frame_rate")),
        "r_fps": ratio(s.get("r_frame_rate")),
        "frame_count": int(s["nb_frames"]) if s.get("nb_frames") else None,
        "duration_s": float(s["duration"]) if s.get("duration") else None,
    }


def frame_timestamps(path: str | Path) -> list[float]:
    """Return editor-style relative timestamps for each decoded frame.

    We subtract the first frame's container timestamp so CVAT frame 0 maps to t=0.
    """
    path = Path(path)
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0", str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    ts = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split(",", 1)[0]
        try:
            ts.append(float(token))
        except ValueError:
            continue
    if not ts:
        raise ValueError(f"No frame timestamps found in {path}")
    t0 = ts[0]
    return [t - t0 for t in ts]


def load_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
