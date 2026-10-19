from __future__ import annotations

import pandas as pd


def stage_qc(stages: pd.DataFrame, video_duration_s: float) -> pd.DataFrame:
    rows = []
    prev_end = None
    for _, s in stages.iterrows():
        rows.append({
            "stage_id": s.stage_id,
            "stage_type": s.stage_type,
            "duration_s": s.duration_s,
            "positive_duration": bool(s.duration_s > 0),
            "within_video": bool(s.start_s >= 0 and s.end_s <= video_duration_s + 0.2),
            "gap_from_previous_s": None if prev_end is None else s.start_s - prev_end,
        })
        prev_end = s.end_s
    return pd.DataFrame(rows)


def gaze_qc(fused: pd.DataFrame) -> dict:
    n = len(fused)
    valid = fused["gaze_valid"].fillna(False)
    overall = float(valid.mean()) if n else None
    by_stage = []
    for st, g in fused[fused["stage_type"].notna()].groupby("stage_type", dropna=False):
        by_stage.append({
            "stage_type": st,
            "samples": len(g),
            "valid_gaze_ratio": float(g["gaze_valid"].fillna(False).mean()),
        })
    return {"samples": n, "valid_gaze_ratio": overall, "by_stage": by_stage}
