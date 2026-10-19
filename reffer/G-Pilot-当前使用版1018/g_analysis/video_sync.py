from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_frame_indices(times_s: pd.Series, frame_ts: list[float]) -> pd.Series:
    arr = np.asarray(frame_ts, dtype=float)
    t = pd.to_numeric(times_s, errors="coerce").to_numpy(dtype=float)
    idx = np.searchsorted(arr, t, side="left")
    idx = np.clip(idx, 0, len(arr)-1)
    prev = np.maximum(idx - 1, 0)
    choose_prev = np.abs(t - arr[prev]) <= np.abs(t - arr[idx])
    idx = np.where(choose_prev, prev, idx)
    return pd.Series(idx, index=times_s.index, dtype="Int64")
