from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# Engineering QC thresholds for pipeline verification.
# These are NOT research thresholds and should not be reported as substantive findings.
PASS_CORR = 0.98
WARN_CORR = 0.95
PASS_DIRECTION = 0.98
WARN_DIRECTION = 0.95
PASS_OUTSIDE_DIRECTION = 0.95
WARN_OUTSIDE_DIRECTION = 0.90
PASS_CENTER_OFFSET_DEG = 3.0
WARN_CENTER_OFFSET_DEG = 5.0
PASS_Z_POSITIVE = 0.99
WARN_Z_POSITIVE = 0.95


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


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if int(m.sum()) < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _direction_agreement(x: pd.Series, h: pd.Series, axis: str) -> tuple[float, int]:
    """Check whether normalized scene position and H/V point in the same direction.

    H convention: right positive, so X > 0.5 should generally imply H > 0.
    V convention: up positive, while image Y grows downward, so Y < 0.5 should imply V > 0.

    Near the center is excluded because sign is unstable around zero.
    """
    if axis == "h":
        m = x.notna() & h.notna() & ((x - 0.5).abs() > 0.02) & (h.abs() > 1.0)
        if int(m.sum()) == 0:
            return np.nan, 0
        ok = ((x[m] - 0.5) * h[m]) > 0
    else:
        m = x.notna() & h.notna() & ((x - 0.5).abs() > 0.02) & (h.abs() > 1.0)
        if int(m.sum()) == 0:
            return np.nan, 0
        ok = ((0.5 - x[m]) * h[m]) > 0

    return float(ok.mean()), int(m.sum())


def _outside_direction_agreement(
    gx: pd.Series, gy: pd.Series, h: pd.Series, v: pd.Series
) -> tuple[float, int]:
    """For gaze projected outside the scene frame, verify directional consistency.

    Examples:
      scene X < 0 -> gaze should be left -> H < 0
      scene X > 1 -> gaze should be right -> H > 0
      scene Y < 0 -> gaze should be above -> V > 0
      scene Y > 1 -> gaze should be below -> V < 0
    """
    results = []

    for x, y, hh, vv in zip(gx, gy, h, v):
        checks = []
        if pd.notna(x) and pd.notna(hh):
            if x < 0:
                checks.append(hh < 0)
            elif x > 1:
                checks.append(hh > 0)

        if pd.notna(y) and pd.notna(vv):
            if y < 0:
                checks.append(vv > 0)
            elif y > 1:
                checks.append(vv < 0)

        if checks:
            results.append(all(checks))

    if not results:
        return np.nan, 0
    return float(np.mean(results)), len(results)


def _center_median(scene_coord: pd.Series, angle: pd.Series, band: float = 0.025) -> tuple[float, int]:
    m = scene_coord.notna() & angle.notna() & ((scene_coord - 0.5).abs() <= band)
    if int(m.sum()) == 0:
        return np.nan, 0
    return float(angle[m].median()), int(m.sum())


def _grade_metric(value: float, pass_rule, warn_rule) -> str:
    if pd.isna(value):
        return "NA"
    if pass_rule(value):
        return "PASS"
    if warn_rule(value):
        return "WARN"
    return "FAIL"


def _overall_status(grades: list[str]) -> str:
    grades = [g for g in grades if g != "NA"]
    if not grades:
        return "WARN"
    if "FAIL" in grades:
        return "FAIL"
    if "WARN" in grades:
        return "WARN"
    return "PASS"


def _load_settings(root: Path) -> dict:
    path = root / "config" / "analysis_settings.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _make_plots(df: pd.DataFrame, outdir: Path, dataset_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    for font in ['PingFang SC','Heiti SC','Microsoft YaHei','SimHei','Noto Sans CJK SC','Arial Unicode MS']:
        if font in available:
            plt.rcParams['font.sans-serif']=[font,'DejaVu Sans'];break
    plt.rcParams['axes.unicode_minus']=False

    outdir.mkdir(parents=True, exist_ok=True)

    gx = _num(df, "GazePointX")
    gy = _num(df, "GazePointY")
    h = _num(df, "horizontal_visual_angle_deg")
    v = _num(df, "vertical_visual_angle_deg")

    scene_inside = gx.between(0, 1) & gy.between(0, 1)

    # Scene X -> H
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(gx[scene_inside], h[scene_inside], s=5, alpha=0.28, label="scene-frame gaze")
    ax.scatter(gx[~scene_inside], h[~scene_inside], s=8, alpha=0.45, label="projected outside scene frame")
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.axvline(1.0, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Tobii scene GazePointX (normalized)")
    ax.set_ylabel("Horizontal visual angle H (deg; right +)")
    ax.set_title(f"{dataset_name} | Scene X vs H")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "scene_x_vs_h.png", dpi=180)
    plt.close(fig)

    # Scene Y -> V
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(gy[scene_inside], v[scene_inside], s=5, alpha=0.28, label="scene-frame gaze")
    ax.scatter(gy[~scene_inside], v[~scene_inside], s=8, alpha=0.45, label="projected outside scene frame")
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.axvline(1.0, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Tobii scene GazePointY (normalized; down +)")
    ax.set_ylabel("Vertical visual angle V (deg; up +)")
    ax.set_title(f"{dataset_name} | Scene Y vs V")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "scene_y_vs_v.png", dpi=180)
    plt.close(fig)

    # H/V map: scene-inside vs projected-outside
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.scatter(h[scene_inside], v[scene_inside], s=6, alpha=0.25, label="scene-frame gaze")
    ax.scatter(h[~scene_inside], v[~scene_inside], s=10, alpha=0.55, label="projected outside scene frame")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Horizontal visual angle H (deg; right +)")
    ax.set_ylabel("Vertical visual angle V (deg; up +)")
    ax.set_title(f"{dataset_name} | H/V coordinate QC")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "hv_scene_inside_vs_outside.png", dpi=180)
    plt.close(fig)


def inspect_dataset(folder: Path, settings: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    preferred = folder / "gaze_stage_aoi_fused_v03.csv"
    fallback = folder / "gaze_stage_aoi_fused.csv"
    path = preferred if preferred.exists() else fallback

    df = pd.read_csv(path)

    required = [
        "GazePointX", "GazePointY",
        "horizontal_visual_angle_deg", "vertical_visual_angle_deg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    valid = _bool_series(df["gaze_valid"]) if "gaze_valid" in df.columns else pd.Series(True, index=df.index)

    gx = _num(df, "GazePointX")
    gy = _num(df, "GazePointY")
    h = _num(df, "horizontal_visual_angle_deg")
    v = _num(df, "vertical_visual_angle_deg")
    z = _num(df, "GazePositionZ")

    usable = valid & gx.notna() & gy.notna() & h.notna() & v.notna()
    q = df.loc[usable].copy()

    gx = gx[usable]
    gy = gy[usable]
    h = h[usable]
    v = v[usable]
    z = z[usable]

    scene_inside = gx.between(0, 1) & gy.between(0, 1)
    scene_outside = ~scene_inside

    corr_x_h = _safe_corr(gx, h)
    corr_y_v = _safe_corr(gy, v)

    h_dir, h_dir_n = _direction_agreement(gx, h, "h")
    v_dir, v_dir_n = _direction_agreement(gy, v, "v")
    out_dir, out_dir_n = _outside_direction_agreement(gx, gy, h, v)

    center_h, center_h_n = _center_median(gx, h)
    center_v, center_v_n = _center_median(gy, v)

    z_positive_ratio = float((z > 0).mean()) if z.notna().any() else np.nan

    # Previous visualization clipping diagnostic only.
    # This does NOT claim the camera's optical FOV is identical to these bounds.
    hfov = settings.get("scene_hfov_deg", np.nan)
    vfov = settings.get("scene_vfov_deg", np.nan)

    if pd.notna(hfov) and pd.notna(vfov):
        old_plot_inside = (
            h.between(-float(hfov)/2, float(hfov)/2)
            & v.between(-float(vfov)/2, float(vfov)/2)
        )
        legacy_plot_outside_ratio = float((~old_plot_inside).mean())
    else:
        legacy_plot_outside_ratio = np.nan

    grades = {
        "corr_x_h": _grade_metric(
            corr_x_h,
            lambda x: x >= PASS_CORR,
            lambda x: x >= WARN_CORR,
        ),
        "corr_y_v": _grade_metric(
            corr_y_v,
            lambda x: x <= -PASS_CORR,
            lambda x: x <= -WARN_CORR,
        ),
        "h_direction": _grade_metric(
            h_dir,
            lambda x: x >= PASS_DIRECTION,
            lambda x: x >= WARN_DIRECTION,
        ),
        "v_direction": _grade_metric(
            v_dir,
            lambda x: x >= PASS_DIRECTION,
            lambda x: x >= WARN_DIRECTION,
        ),
        "outside_direction": _grade_metric(
            out_dir,
            lambda x: x >= PASS_OUTSIDE_DIRECTION,
            lambda x: x >= WARN_OUTSIDE_DIRECTION,
        ),
        "center_h": _grade_metric(
            abs(center_h) if pd.notna(center_h) else np.nan,
            lambda x: x <= PASS_CENTER_OFFSET_DEG,
            lambda x: x <= WARN_CENTER_OFFSET_DEG,
        ),
        "center_v": _grade_metric(
            abs(center_v) if pd.notna(center_v) else np.nan,
            lambda x: x <= PASS_CENTER_OFFSET_DEG,
            lambda x: x <= WARN_CENTER_OFFSET_DEG,
        ),
        "z_positive": _grade_metric(
            z_positive_ratio,
            lambda x: x >= PASS_Z_POSITIVE,
            lambda x: x >= WARN_Z_POSITIVE,
        ),
    }

    status = _overall_status(list(grades.values()))

    participant = str(q["participant_id"].iloc[0]) if "participant_id" in q.columns and len(q) else folder.name.split("__")[0]
    task = str(q["task_id"].iloc[0]) if "task_id" in q.columns and len(q) else (folder.name.split("__")[1] if "__" in folder.name else "")

    summary = {
        "dataset": folder.name,
        "participant_id": participant,
        "task_id": task,
        "source_file": path.name,
        "usable_valid_samples": int(len(q)),
        "scene_inside_samples": int(scene_inside.sum()),
        "scene_outside_samples": int(scene_outside.sum()),
        "scene_outside_ratio": float(scene_outside.mean()) if len(scene_outside) else np.nan,
        "corr_scene_x_vs_H": corr_x_h,
        "corr_scene_y_vs_V": corr_y_v,
        "H_direction_agreement": h_dir,
        "H_direction_n": h_dir_n,
        "V_direction_agreement": v_dir,
        "V_direction_n": v_dir_n,
        "outside_projection_direction_agreement": out_dir,
        "outside_projection_n": out_dir_n,
        "center_H_median_deg": center_h,
        "center_H_n": center_h_n,
        "center_V_median_deg": center_v,
        "center_V_n": center_v_n,
        "GazePositionZ_positive_ratio": z_positive_ratio,
        "H_min_deg": float(h.min()) if len(h) else np.nan,
        "H_max_deg": float(h.max()) if len(h) else np.nan,
        "V_min_deg": float(v.min()) if len(v) else np.nan,
        "V_max_deg": float(v.max()) if len(v) else np.nan,
        "legacy_plot_outside_ratio": legacy_plot_outside_ratio,
        "grade_corr_x_h": grades["corr_x_h"],
        "grade_corr_y_v": grades["corr_y_v"],
        "grade_H_direction": grades["h_direction"],
        "grade_V_direction": grades["v_direction"],
        "grade_outside_direction": grades["outside_direction"],
        "grade_center_H": grades["center_h"],
        "grade_center_V": grades["center_v"],
        "grade_Z_positive": grades["z_positive"],
        "qc_status": status,
    }

    # Annotated samples for diagnosis.
    annotated = q.copy()
    annotated["scene_inside_frame"] = scene_inside.values
    annotated["scene_outside_frame"] = scene_outside.values

    reasons = []
    for x, y in zip(gx, gy):
        r = []
        if x < 0:
            r.append("left")
        elif x > 1:
            r.append("right")
        if y < 0:
            r.append("top")
        elif y > 1:
            r.append("bottom")
        reasons.append("|".join(r))
    annotated["scene_outside_direction"] = reasons

    # Stage-level diagnostics.
    stage_rows = []
    if "stage_type" in annotated.columns:
        use_stage = annotated[annotated["stage_type"].notna()].copy()
        for stage, g in use_stage.groupby("stage_type", sort=False):
            n = len(g)
            out_n = int(g["scene_outside_frame"].sum())
            stage_rows.append({
                "dataset": folder.name,
                "participant_id": participant,
                "task_id": task,
                "stage_type": stage,
                "usable_valid_samples": int(n),
                "scene_outside_samples": out_n,
                "scene_outside_ratio": float(out_n / n) if n else np.nan,
                "H_min_deg": float(pd.to_numeric(g["horizontal_visual_angle_deg"], errors="coerce").min()),
                "H_max_deg": float(pd.to_numeric(g["horizontal_visual_angle_deg"], errors="coerce").max()),
                "V_min_deg": float(pd.to_numeric(g["vertical_visual_angle_deg"], errors="coerce").min()),
                "V_max_deg": float(pd.to_numeric(g["vertical_visual_angle_deg"], errors="coerce").max()),
            })

    return summary, annotated, pd.DataFrame(stage_rows)


def main():
    root = Path(__file__).resolve().parent
    output_root = root / "output"
    qc_root = output_root / "_spatial_coordinate_qc"
    qc_root.mkdir(parents=True, exist_ok=True)

    settings = _load_settings(root)

    folders = [
        p for p in sorted(output_root.iterdir())
        if p.is_dir()
        and not p.name.startswith("_")
        and (
            (p / "gaze_stage_aoi_fused_v03.csv").exists()
            or (p / "gaze_stage_aoi_fused.csv").exists()
        )
    ]

    if not folders:
        print("[停止] output/ 下没有可用于空间QC的数据包。")
        raise SystemExit(1)

    summaries = []
    stage_tables = []

    print("\n=== G-line V0.3.2 spatial_coordinate_qc ===")
    print("目的：验证 Tobii scene gaze 与 H/V 坐标转换是否方向一致、中心合理，")
    print("并判断超出 scene frame 的样本是否是方向一致的外投影，而不是盲目扩大画布。\n")

    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] {folder.name}")

        try:
            summary, annotated, stage_table = inspect_dataset(folder, settings)
        except Exception as e:
            print(f"  [FAIL] {e}")
            summaries.append({
                "dataset": folder.name,
                "qc_status": "FAIL",
                "error": str(e),
            })
            continue

        ds_out = qc_root / folder.name
        ds_out.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(ds_out / "spatial_coordinate_qc_summary.csv", index=False, encoding="utf-8-sig")
        annotated.to_csv(ds_out / "spatial_coordinate_qc_samples.csv", index=False, encoding="utf-8-sig")
        stage_table.to_csv(ds_out / "outside_scene_by_stage.csv", index=False, encoding="utf-8-sig")

        _make_plots(annotated, ds_out / "figures", folder.name)

        summaries.append(summary)
        if len(stage_table):
            stage_tables.append(stage_table)

        print(
            f"  [{summary['qc_status']}] "
            f"corr X-H={summary['corr_scene_x_vs_H']:.4f}; "
            f"corr Y-V={summary['corr_scene_y_vs_V']:.4f}; "
            f"scene-outside={summary['scene_outside_ratio']:.3%}"
        )

    all_summary = pd.DataFrame(summaries)
    all_summary.to_csv(
        qc_root / "spatial_coordinate_qc_all_datasets.csv",
        index=False, encoding="utf-8-sig"
    )

    if stage_tables:
        pd.concat(stage_tables, ignore_index=True).to_csv(
            qc_root / "outside_scene_all_stages.csv",
            index=False, encoding="utf-8-sig"
        )

    readme = """G-line spatial_coordinate_qc
================================

This module is a technical validation step, not a substantive research analysis.

It checks:
1. Tobii scene GazePointX vs horizontal visual angle H:
   X grows rightward, so the relationship should be strongly positive.
2. Tobii scene GazePointY vs vertical visual angle V:
   image Y grows downward while V uses up-positive, so the relationship should be strongly negative.
3. Direction consistency around the visual center.
4. Samples projected outside the scene frame:
   X<0 / X>1 / Y<0 / Y>1 should agree with the sign of H/V.
5. Near-center gaze should map close to H=0, V=0.
6. GazePositionZ should stay positive for valid forward gaze.

Interpretation:
- scene_inside_frame=True:
  the gaze projection lies inside the scene camera frame.
- scene_outside_frame=True:
  the exported gaze projection lies outside the scene frame.
  If direction consistency is high, these samples should NOT automatically be called errors.
  They indicate gaze directions beyond the scene image boundary and need to be handled explicitly.

Important:
- PASS/WARN/FAIL thresholds in this script are engineering QC rules for pipeline verification.
  They are not literature-derived scientific thresholds.
- Do not use this module to claim a biological human field of view.
- Do not infer OST/F0 display limits from these QC values.
"""

    (qc_root / "README_spatial_coordinate_qc.txt").write_text(readme, encoding="utf-8")

    print("\n=== spatial_coordinate_qc 完成 ===")
    print(f"总结果: {qc_root / 'spatial_coordinate_qc_all_datasets.csv'}")
    print(f"分Stage: {qc_root / 'outside_scene_all_stages.csv'}")
    print("\n下一步不是立刻扩大正式底图。")
    print("先根据QC结果决定：scene-outside样本是合法外投影，还是H/V转换需要修正。")
    from g_analysis.run_quality import result_state
    state, code = result_state(all_summary)
    print(f"空间QC状态：{state}；PASS只表示自动检查通过，不替代同步和AOI人工抽查。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
