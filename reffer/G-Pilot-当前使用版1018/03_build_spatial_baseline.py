from __future__ import annotations

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd


VERSION = "V0.3.3"

# This is only a plotting margin for the current Mini Test / Pilot QC map.
# It is NOT a scientific threshold and NOT the final frozen G-line plot range.
CURRENT_RUN_MARGIN_DEG = 5.0
ROUND_TO_DEG = 5.0


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def _safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))
    return text.strip("_") or "unnamed"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _add_eccentricity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "eccentricity_deg" in out.columns:
        out["eccentricity_deg"] = pd.to_numeric(
            out["eccentricity_deg"], errors="coerce"
        )
        return out

    xyz = {"GazePositionX", "GazePositionY", "GazePositionZ"}
    if xyz.issubset(out.columns):
        x = _num(out, "GazePositionX")
        y = _num(out, "GazePositionY")
        z = _num(out, "GazePositionZ")
        out["eccentricity_deg"] = np.degrees(
            np.arctan2(np.sqrt(x * x + y * y), z)
        )
        return out

    h = _num(out, "horizontal_visual_angle_deg")
    v = _num(out, "vertical_visual_angle_deg")
    hr = np.radians(h)
    vr = np.radians(v)
    out["eccentricity_deg"] = np.degrees(
        np.arctan(np.sqrt(np.tan(hr) ** 2 + np.tan(vr) ** 2))
    )
    return out


def _weighted_quantile(values, weights, qs):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return [np.nan for _ in qs]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cw = np.cumsum(weights)
    cw = cw / cw[-1]
    return [float(np.interp(q, cw, values)) for q in qs]


def _participant_stage_summary(spatial: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (participant, task, stage), g in spatial.groupby(
        ["participant_id", "task_id", "stage_type"],
        dropna=False,
        sort=False,
    ):
        e = _num(g, "eccentricity_deg").dropna()
        h = _num(g, "horizontal_visual_angle_deg").dropna()
        v = _num(g, "vertical_visual_angle_deg").dropna()

        rows.append(
            {
                "participant_id": participant,
                "task_id": task,
                "stage_type": stage,
                "valid_spatial_samples": int(len(g)),
                "h_median_deg": float(h.median()) if len(h) else np.nan,
                "v_median_deg": float(v.median()) if len(v) else np.nan,
                "eccentricity_median_deg": float(e.median()) if len(e) else np.nan,
                "eccentricity_p25_deg": float(e.quantile(0.25)) if len(e) else np.nan,
                "eccentricity_p75_deg": float(e.quantile(0.75)) if len(e) else np.nan,
                "eccentricity_p90_deg": float(e.quantile(0.90)) if len(e) else np.nan,
                "eccentricity_p95_deg": float(e.quantile(0.95)) if len(e) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _group_stage_summary(spatial: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (task, stage), g in spatial.groupby(
        ["task_id", "stage_type"], dropna=False, sort=False
    ):
        e = _num(g, "eccentricity_deg")
        w = _num(g, "group_weight")

        q25, q50, q75, q90, q95 = _weighted_quantile(
            e, w, [0.25, 0.50, 0.75, 0.90, 0.95]
        )

        rows.append(
            {
                "task_id": task,
                "stage_type": stage,
                "participant_count": int(g["participant_id"].nunique()),
                "valid_spatial_samples": int(len(g)),
                "group_eccentricity_p25_deg": q25,
                "group_eccentricity_median_deg": q50,
                "group_eccentricity_p75_deg": q75,
                "group_eccentricity_p90_deg": q90,
                "group_eccentricity_p95_deg": q95,
                "weight_sum_check": float(w.sum()),
            }
        )

    return pd.DataFrame(rows)


def _scene_projection_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gx = _num(out, "GazePointX")
    gy = _num(out, "GazePointY")

    if gx.notna().any() and gy.notna().any():
        out["scene_inside_frame"] = gx.between(0, 1) & gy.between(0, 1)
        out["scene_outside_frame"] = ~out["scene_inside_frame"]
    else:
        out["scene_inside_frame"] = pd.NA
        out["scene_outside_frame"] = pd.NA

    return out


def _derive_current_run_range(spatial: pd.DataFrame) -> dict:
    """Make one common plot range for the current run.

    IMPORTANT:
    - Includes every valid H/V sample in the current run.
    - Symmetric around H=0 and V=0.
    - Adds a small engineering margin and rounds outward.
    - This is a QC/display range only, NOT the final formal G-line range.
    """
    h = _num(spatial, "horizontal_visual_angle_deg").dropna()
    v = _num(spatial, "vertical_visual_angle_deg").dropna()

    if len(h) == 0 or len(v) == 0:
        raise ValueError("No valid H/V values available for current-run plot range.")

    h_abs = max(abs(float(h.min())), abs(float(h.max())))
    v_abs = max(abs(float(v.min())), abs(float(v.max())))

    h_limit = math.ceil((h_abs + CURRENT_RUN_MARGIN_DEG) / ROUND_TO_DEG) * ROUND_TO_DEG
    v_limit = math.ceil((v_abs + CURRENT_RUN_MARGIN_DEG) / ROUND_TO_DEG) * ROUND_TO_DEG

    return {
        "h_min_deg": -float(h_limit),
        "h_max_deg": float(h_limit),
        "v_min_deg": -float(v_limit),
        "v_max_deg": float(v_limit),
        "derivation": (
            "current-run all-valid H/V extrema, symmetric around zero, "
            f"+{CURRENT_RUN_MARGIN_DEG:.1f} deg margin, rounded outward to "
            f"{ROUND_TO_DEG:.1f} deg"
        ),
        "status": "PROVISIONAL_MINITEST_OR_PILOT_QC_RANGE_NOT_FORMAL_FROZEN_RANGE",
    }


def _spatial_range_qc(
    spatial: pd.DataFrame,
    current_range: dict,
    scene_hfov_deg: float | None,
    scene_vfov_deg: float | None,
) -> pd.DataFrame:
    rows = []

    for (task, stage), g in spatial.groupby(
        ["task_id", "stage_type"], dropna=False, sort=False
    ):
        h = _num(g, "horizontal_visual_angle_deg")
        v = _num(g, "vertical_visual_angle_deg")
        w = _num(g, "group_weight")

        hq = _weighted_quantile(h, w, [0.005, 0.01, 0.50, 0.99, 0.995])
        vq = _weighted_quantile(v, w, [0.005, 0.01, 0.50, 0.99, 0.995])

        scene_outside_mass = np.nan
        if "scene_outside_frame" in g.columns:
            s = g["scene_outside_frame"]
            known = s.notna()
            if known.any():
                scene_outside_mass = float(w[known & (s == True)].sum())

        legacy_rect_outside_mass = np.nan
        if scene_hfov_deg is not None and scene_vfov_deg is not None:
            legacy_inside = (
                h.between(-scene_hfov_deg / 2.0, scene_hfov_deg / 2.0)
                & v.between(-scene_vfov_deg / 2.0, scene_vfov_deg / 2.0)
            )
            legacy_rect_outside_mass = float(w[~legacy_inside].sum())

        current_inside = (
            h.between(current_range["h_min_deg"], current_range["h_max_deg"])
            & v.between(current_range["v_min_deg"], current_range["v_max_deg"])
        )
        current_run_clipped_mass = float(w[~current_inside].sum())

        rows.append(
            {
                "task_id": task,
                "stage_type": stage,
                "participant_count": int(g["participant_id"].nunique()),
                "weight_sum_check": float(w.sum()),
                "H_min_deg": float(h.min()),
                "H_p0_5_deg": hq[0],
                "H_p1_deg": hq[1],
                "H_median_deg": hq[2],
                "H_p99_deg": hq[3],
                "H_p99_5_deg": hq[4],
                "H_max_deg": float(h.max()),
                "V_min_deg": float(v.min()),
                "V_p0_5_deg": vq[0],
                "V_p1_deg": vq[1],
                "V_median_deg": vq[2],
                "V_p99_deg": vq[3],
                "V_p99_5_deg": vq[4],
                "V_max_deg": float(v.max()),
                "scene_projection_outside_mass": scene_outside_mass,
                "legacy_scene_fov_rect_outside_mass": legacy_rect_outside_mass,
                "current_run_plot_clipped_mass": current_run_clipped_mass,
            }
        )

    return pd.DataFrame(rows)


def _make_group_maps(
    spatial: pd.DataFrame,
    figdir: Path,
    current_range: dict,
    scene_hfov_deg: float | None,
    scene_vfov_deg: float | None,
):
    """Create full-data group baseline figures.

    Hexagons are visualization only.
    Future F0/A2 occupancy_mass will be calculated from group_weighted H/V samples,
    not from the displayed hexagons.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    full_dir = figdir / "full_group_baseline"
    reference_dir = figdir / "with_scene_camera_reference"
    full_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    extent = [
        current_range["h_min_deg"],
        current_range["h_max_deg"],
        current_range["v_min_deg"],
        current_range["v_max_deg"],
    ]

    for (task, stage), g in spatial.groupby(
        ["task_id", "stage_type"], dropna=False, sort=False
    ):
        h = _num(g, "horizontal_visual_angle_deg")
        v = _num(g, "vertical_visual_angle_deg")
        w = _num(g, "group_weight")

        m = h.notna() & v.notna() & w.notna()
        h = h[m]
        v = v[m]
        w = w[m]

        # A. Full standard-coordinate baseline, no scene-camera clipping.
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        if len(h):
            hb = ax.hexbin(
                h,
                v,
                C=w * 100.0,
                reduce_C_function=np.sum,
                gridsize=50,
                mincnt=1,
                extent=extent,
            )
            fig.colorbar(
                hb,
                ax=ax,
                label="Group occupancy mass per display hex (%)",
            )

        ax.axhline(0, linewidth=0.8, linestyle="--")
        ax.axvline(0, linewidth=0.8, linestyle="--")
        ax.set_xlim(current_range["h_min_deg"], current_range["h_max_deg"])
        ax.set_ylim(current_range["v_min_deg"], current_range["v_max_deg"])
        ax.set_xlabel("Horizontal visual angle H (deg; right +)")
        ax.set_ylabel("Vertical visual angle V (deg; up +)")
        ax.set_title(
            f"{task} | {stage} | full group visual baseline | "
            f"n={g['participant_id'].nunique()}"
        )
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        fig.savefig(
            full_dir / f"{_safe_name(task)}__{_safe_name(stage)}__full_group_baseline.png",
            dpi=180,
        )
        plt.close(fig)

        # B. Same exact map, with old scene-camera FOV rectangle as a visual reference only.
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        if len(h):
            hb = ax.hexbin(
                h,
                v,
                C=w * 100.0,
                reduce_C_function=np.sum,
                gridsize=50,
                mincnt=1,
                extent=extent,
            )
            fig.colorbar(
                hb,
                ax=ax,
                label="Group occupancy mass per display hex (%)",
            )

        ax.axhline(0, linewidth=0.8, linestyle="--")
        ax.axvline(0, linewidth=0.8, linestyle="--")

        if scene_hfov_deg is not None and scene_vfov_deg is not None:
            rect = Rectangle(
                (-scene_hfov_deg / 2.0, -scene_vfov_deg / 2.0),
                scene_hfov_deg,
                scene_vfov_deg,
                fill=False,
                linewidth=1.2,
                linestyle=":",
                label=(
                    f"scene camera FOV reference "
                    f"({scene_hfov_deg:g}°×{scene_vfov_deg:g}°; not validity boundary)"
                ),
            )
            ax.add_patch(rect)
            ax.legend(loc="best")

        ax.set_xlim(current_range["h_min_deg"], current_range["h_max_deg"])
        ax.set_ylim(current_range["v_min_deg"], current_range["v_max_deg"])
        ax.set_xlabel("Horizontal visual angle H (deg; right +)")
        ax.set_ylabel("Vertical visual angle V (deg; up +)")
        ax.set_title(
            f"{task} | {stage} | group baseline + scene-camera reference | "
            f"n={g['participant_id'].nunique()}"
        )
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        fig.savefig(
            reference_dir
            / f"{_safe_name(task)}__{_safe_name(stage)}__scene_reference.png",
            dpi=180,
        )
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    output_root = root / "output"

    # Write to a NEW versioned folder so V0.3 remains available for comparison.
    group_root = output_root / "_group_spatial_v033"
    group_root.mkdir(parents=True, exist_ok=True)

    settings_path = root / "config" / "analysis_settings.json"
    try:
        settings = (
            json.loads(settings_path.read_text(encoding="utf-8"))
            if settings_path.exists()
            else {}
        )
    except Exception:
        settings = {}

    scene_hfov = settings.get("scene_hfov_deg", 95.0)
    scene_vfov = settings.get("scene_vfov_deg", 63.0)

    try:
        scene_hfov = float(scene_hfov)
        scene_vfov = float(scene_vfov)
    except Exception:
        scene_hfov = None
        scene_vfov = None

    from g_analysis.run_quality import eligible_datasets
    eligible = eligible_datasets(output_root)
    datasets = []
    for folder in sorted(output_root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith("_") or folder.name.startswith(".") or folder.name not in eligible:
            continue

        preferred = folder / "gaze_stage_aoi_fused_v03.csv"
        fallback = folder / "gaze_stage_aoi_fused.csv"

        if preferred.exists() or fallback.exists():
            datasets.append(folder)

    if not datasets:
        print("[停止] 没找到 output/被试__任务/ 的 fused gaze 文件。")
        raise SystemExit(1)

    all_spatial = []

    print(f"\n=== G-line {VERSION} 标准化群体底图修正 ===")
    print("本版不再用 scene camera FOV 裁剪 G 的正式 H/V 数据。")
    print("Mini Test 当前图域只用于完整显示当前数据，不是正式冻结边界。")
    print("scene camera FOV 只作为独立参考层保留。\n")

    for i, folder in enumerate(datasets, 1):
        preferred = folder / "gaze_stage_aoi_fused_v03.csv"
        fallback = folder / "gaze_stage_aoi_fused.csv"
        path = preferred if preferred.exists() else fallback

        fused = pd.read_csv(path)
        fused = _add_eccentricity(fused)
        fused = _scene_projection_flags(fused)

        valid = (
            _bool_series(fused["spatial_valid"] if "spatial_valid" in fused else fused["gaze_valid"])
            if "gaze_valid" in fused.columns
            else pd.Series(False, index=fused.index)
        )

        keep_cols = [
            "participant_id",
            "task_id",
            "sample_id",
            "video_time_s",
            "stage_id",
            "stage_type",
            "stage_occurrence",
            "horizontal_visual_angle_deg",
            "vertical_visual_angle_deg",
            "eccentricity_deg",
            "GazePointX",
            "GazePointY",
            "scene_inside_frame",
            "scene_outside_frame",
        ]
        keep_cols = [c for c in keep_cols if c in fused.columns]

        spatial = fused.loc[
            valid
            & fused["stage_type"].notna()
            & _num(fused, "horizontal_visual_angle_deg").notna()
            & _num(fused, "vertical_visual_angle_deg").notna()
            & _num(fused, "eccentricity_deg").notna(),
            keep_cols,
        ].copy()

        counts = spatial.groupby(
            ["participant_id", "task_id", "stage_type"],
            dropna=False,
        )["sample_id"].transform("count")

        spatial["participant_stage_weight"] = 1.0 / counts
        spatial["source_dataset"] = folder.name

        all_spatial.append(spatial)

        spatial.to_csv(
            folder / "spatial_samples_v033.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"[{i}/{len(datasets)}] {folder.name}: "
            f"valid spatial samples={len(spatial)}"
        )

    combined = pd.concat(all_spatial, ignore_index=True)

    # Equal participant weighting within each Task × Stage.
    n_participants = combined.groupby(
        ["task_id", "stage_type"],
        dropna=False,
    )["participant_id"].transform("nunique")

    combined["group_weight"] = (
        combined["participant_stage_weight"] / n_participants
    )

    # Current-run full range: one identical coordinate range for every Task/Stage.
    current_range = _derive_current_run_range(combined)

    # Core group sample table for future F0/A2 occupancy_mass.
    combined.to_csv(
        group_root / "group_spatial_samples_v033.csv",
        index=False,
        encoding="utf-8-sig",
    )

    participant_summary = _participant_stage_summary(combined)
    group_summary = _group_stage_summary(combined)
    range_qc = _spatial_range_qc(
        combined,
        current_range,
        scene_hfov,
        scene_vfov,
    )

    participant_summary.to_csv(
        group_root / "eccentricity_participant_stage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_summary.to_csv(
        group_root / "eccentricity_group_stage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    range_qc.to_csv(
        group_root / "spatial_range_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (group_root / "current_run_plot_range.json").write_text(
        json.dumps(current_range, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _make_group_maps(
        combined,
        group_root / "figures",
        current_range,
        scene_hfov,
        scene_vfov,
    )

    readme = f"""G-line {VERSION} group spatial baseline
============================================

Purpose
-------
This version corrects the V0.3 visualization logic after spatial-coordinate QC.

1. G-line valid H/V gaze samples are NOT clipped by the scene-camera FOV.
2. All Task × Stage maps in this run use one identical H/V coordinate range.
3. The current-run range is derived only to display all current Mini Test / Pilot data.
   It is NOT the final frozen formal-study range.
4. Scene camera FOV ({scene_hfov if scene_hfov is not None else 'NA'} deg ×
   {scene_vfov if scene_vfov is not None else 'NA'} deg) is retained only as a visual reference.
5. Future OST/F0 usable display window must be added as a separate layer.
6. Hexagons remain visualization only. Future occupancy_mass will be calculated
   directly from group_weighted H/V samples.

Current run plot range
----------------------
H: {current_range['h_min_deg']} to {current_range['h_max_deg']} deg
V: {current_range['v_min_deg']} to {current_range['v_max_deg']} deg

Important QC outputs
--------------------
- spatial_range_qc.csv
  - scene_projection_outside_mass:
    group occupancy mass whose normalized scene projection is outside [0,1]×[0,1]
  - legacy_scene_fov_rect_outside_mass:
    mass outside the OLD ±scene-FOV/2 display rectangle; comparison only
  - current_run_plot_clipped_mass:
    should be 0 because this QC/full map is designed to display all valid current-run H/V data

- current_run_plot_range.json
  stores the provisional Mini Test/Pilot display range.

Formal range decision
---------------------
Do NOT freeze the formal G-line coordinate display boundary from this Mini Test.
After the 5–6 participant Pilot, use a pre-defined robust rule and then freeze
one common H/V range before formal data collection / final reporting.
"""
    (group_root / "README_v033.txt").write_text(readme, encoding="utf-8")

    print("\n=== V0.3.3 完成 ===")
    print(f"输出目录: {group_root}")
    print("重点检查:")
    print("  1) spatial_range_qc.csv")
    print("  2) current_run_plot_range.json")
    print("  3) figures/full_group_baseline/")
    print("  4) figures/with_scene_camera_reference/")
    print("\n注意：当前范围仍是 Mini Test / Pilot QC 范围，不是正式研究冻结范围。")


if __name__ == "__main__":
    main()
