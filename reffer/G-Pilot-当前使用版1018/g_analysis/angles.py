from __future__ import annotations

import numpy as np
import pandas as pd


def add_head_centered_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head/camera-centered visual-angle variables from exported 3D gaze position.

    Convention:
      horizontal_visual_angle_deg: right positive, left negative
      vertical_visual_angle_deg: up positive, down negative
      eccentricity_deg: angular distance from the head/camera forward axis (0 deg)

    ErgoLAB/Tobii export used here has +X to camera-left, +Z forward;
    therefore horizontal is negated. Eccentricity is computed directly from
    the 3D gaze vector, not approximated from sqrt(H^2 + V^2).
    """
    out = df.copy()

    x = pd.to_numeric(out["GazePositionX"], errors="coerce")
    y = pd.to_numeric(out["GazePositionY"], errors="coerce")
    z = pd.to_numeric(out["GazePositionZ"], errors="coerce")

    out["horizontal_visual_angle_deg"] = -np.degrees(np.arctan2(x, z))
    out["vertical_visual_angle_deg"] = np.degrees(np.arctan2(y, z))

    radial_xy = np.sqrt(x * x + y * y)
    out["eccentricity_deg"] = np.degrees(np.arctan2(radial_xy, z))

    return out
