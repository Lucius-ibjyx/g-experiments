from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


GAPS = [0.05, 0.10, 0.15, 0.20]
CURRENT_GAP = 0.10


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _sample_period_s(fused: pd.DataFrame) -> float:
    t = _num(fused["video_time_s"]).dropna().sort_values()
    d = t.diff().dropna()
    d = d[(d > 0) & (d < 0.1)]
    return float(d.median()) if len(d) else 1.0 / 70.0


def _stage_safe_dedup_hits(hits: pd.DataFrame, fused: pd.DataFrame) -> pd.DataFrame:
    stage_cols = [
        c for c in [
            "sample_id", "participant_id", "task_id",
            "stage_id", "stage_type", "stage_occurrence"
        ]
        if c in fused.columns
    ]
    h = hits.merge(
        fused[stage_cols].drop_duplicates("sample_id"),
        on="sample_id",
        how="left",
        suffixes=("", "_fused"),
    )

    for c in ["participant_id", "task_id"]:
        cf = f"{c}_fused"
        if cf in h.columns:
            h[c] = h[cf].combine_first(h.get(c))
            h = h.drop(columns=[cf])

    # Critical rule: only Stage-assigned samples enter formal sensitivity analysis.
    h = h[
        h["stage_id"].notna()
        & h["stage_type"].notna()
        & (h["stage_type"].astype(str).str.strip() != "")
    ].copy()

    h["hit_rank"] = h["hit_type"].map({"exact": 0, "tolerance": 1}).fillna(2)
    if "boundary_distance_deg" in h.columns:
        h["boundary_distance_deg"] = _num(h["boundary_distance_deg"])
    else:
        h["boundary_distance_deg"] = np.nan

    sort_cols = ["sample_id", "semantic_key", "hit_rank", "boundary_distance_deg"]
    if "track_id" in h.columns:
        sort_cols.append("track_id")
    h = h.sort_values(sort_cols)
    h = h.drop_duplicates(["sample_id", "semantic_key"], keep="first")
    return h.drop(columns=["hit_rank"])


def _build_episodes(hits: pd.DataFrame, gap_s: float, sample_period_s: float) -> pd.DataFrame:
    rows = []
    if hits.empty:
        return pd.DataFrame()

    group_cols = ["semantic_key", "stage_id", "stage_type"]
    if "stage_occurrence" in hits.columns:
        group_cols.append("stage_occurrence")

    for _, g in hits.groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values("video_time_s").reset_index(drop=True)

        buf = []
        last_t = None

        def flush():
            nonlocal buf
            if not buf:
                return
            d = pd.DataFrame(buf)
            r0 = d.iloc[0]
            start = float(d["video_time_s"].min())
            end = float(d["video_time_s"].max())
            exact_n = int((d["hit_type"] == "exact").sum())
            tol_n = int((d["hit_type"] == "tolerance").sum())

            rows.append({
                "participant_id": r0.get("participant_id", pd.NA),
                "task_id": r0.get("task_id", pd.NA),
                "stage_id": r0.get("stage_id", pd.NA),
                "stage_type": r0.get("stage_type", pd.NA),
                "semantic_key": r0.get("semantic_key", pd.NA),
                "gap_s": gap_s,
                "start_s": start,
                "end_s": end,
                "duration_s": max(sample_period_s, end - start + sample_period_s),
                "sample_count": int(len(d)),
                "visit_hit_type": "exact" if exact_n > 0 else "tolerance",
            })
            buf = []

        for _, r in g.iterrows():
            t = float(r["video_time_s"])
            if last_t is not None and (t - last_t) > gap_s:
                flush()
            buf.append(r.to_dict())
            last_t = t

        flush()

    return pd.DataFrame(rows)


def _manual_candidates(hits: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    if hits.empty:
        return pd.DataFrame()

    group_cols = ["semantic_key", "stage_id", "stage_type"]
    for _, g in hits.groupby(group_cols, dropna=False, sort=False):
        g = g.sort_values("video_time_s").reset_index(drop=True)
        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            cur = g.iloc[i]
            gap = float(cur["video_time_s"] - prev["video_time_s"])
            if 0.05 < gap <= 0.20:
                rows.append({
                    "dataset": dataset,
                    "participant_id": cur.get("participant_id", pd.NA),
                    "task_id": cur.get("task_id", pd.NA),
                    "stage_id": cur.get("stage_id", pd.NA),
                    "stage_type": cur.get("stage_type", pd.NA),
                    "semantic_key": cur.get("semantic_key", pd.NA),
                    "previous_time_s": float(prev["video_time_s"]),
                    "next_time_s": float(cur["video_time_s"]),
                    "gap_s": gap,
                    "manual_question": "人工上应理解为同一次AOI访问，还是两次？",
                })
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent
    output_root = root / "output"
    audit_root = output_root / "_QC_visit_gap"
    audit_root.mkdir(exist_ok=True)

    dataset_rows = []
    aoi_rows = []
    manual_parts = []

    folders = [
        p for p in sorted(output_root.iterdir())
        if p.is_dir()
        and not p.name.startswith("_")
        and (p / "gaze_stage_aoi_fused.csv").exists()
        and (p / "aoi_hits_effective_long.csv").exists()
    ]

    print("\n=== Visit gap sensitivity — Stage-safe ===")
    print("仅使用已有有效Stage归属的AOI命中。")
    print("比较 gap:", GAPS, "\n")

    for folder in folders:
        fused = pd.read_csv(folder / "gaze_stage_aoi_fused.csv")
        hits = pd.read_csv(folder / "aoi_hits_effective_long.csv")
        sample_period = _sample_period_s(fused)
        safe_hits = _stage_safe_dedup_hits(hits, fused)

        for gap in GAPS:
            ep = _build_episodes(safe_hits, gap, sample_period)
            if ep.empty:
                continue

            duration = _num(ep["duration_s"])
            dataset_rows.append({
                "dataset": folder.name,
                "participant_id": ep["participant_id"].iloc[0],
                "task_id": ep["task_id"].iloc[0],
                "gap_s": gap,
                "visit_count": int(len(ep)),
                "mean_visit_duration_s": float(duration.mean()),
                "median_visit_duration_s": float(duration.median()),
                "single_sample_visit_ratio": float((ep["sample_count"] == 1).mean()),
                "total_hit_samples": int(ep["sample_count"].sum()),
            })

            for (stage, sem), g in ep.groupby(
                ["stage_type", "semantic_key"], dropna=False, sort=False
            ):
                dur = _num(g["duration_s"])
                aoi_rows.append({
                    "dataset": folder.name,
                    "participant_id": g["participant_id"].iloc[0],
                    "task_id": g["task_id"].iloc[0],
                    "stage_type": stage,
                    "semantic_key": sem,
                    "gap_s": gap,
                    "visit_count": int(len(g)),
                    "mean_visit_duration_s": float(dur.mean()),
                    "median_visit_duration_s": float(dur.median()),
                    "total_visit_duration_s": float(dur.sum()),
                    "single_sample_visit_ratio": float((g["sample_count"] == 1).mean()),
                    "total_hit_samples": int(g["sample_count"].sum()),
                })

        m = _manual_candidates(safe_hits, folder.name)
        if len(m):
            manual_parts.append(m)

        print(f"[OK] {folder.name}")

    ds = pd.DataFrame(dataset_rows)
    by_aoi = pd.DataFrame(aoi_rows)
    manual = pd.concat(manual_parts, ignore_index=True) if manual_parts else pd.DataFrame()

    ds.to_csv(
        audit_root / "visit_gap_sensitivity_dataset.csv",
        index=False, encoding="utf-8-sig"
    )
    by_aoi.to_csv(
        audit_root / "visit_gap_sensitivity_by_aoi.csv",
        index=False, encoding="utf-8-sig"
    )
    manual.sort_values("gap_s").to_csv(
        audit_root / "visit_gap_manual_check_candidates.csv",
        index=False, encoding="utf-8-sig"
    )

    if len(ds):
        pivot = ds.pivot(index="dataset", columns="gap_s", values="visit_count")
        if CURRENT_GAP in pivot.columns:
            for gap in GAPS:
                if gap in pivot.columns:
                    pivot[f"change_{gap:.2f}_vs_0.10_pct"] = (
                        (pivot[gap] - pivot[CURRENT_GAP])
                        / pivot[CURRENT_GAP].replace(0, np.nan)
                        * 100
                    )
        pivot.reset_index().to_csv(
            audit_root / "visit_gap_visit_count_comparison.csv",
            index=False, encoding="utf-8-sig"
        )

    readme = """Stage-safe visit gap sensitivity
================================
This version excludes AOI hits without a valid Stage assignment.

Why:
NaN Stage values could previously make every Stage-unassigned sample
look like a new visit because NaN != NaN.

Use this output—not the older sensitivity output—to reassess whether
0.10 s can be frozen for Pilot.
"""
    (audit_root / "README_visit_gap.txt").write_text(readme, encoding="utf-8")

    print("\n完成。请把新的 visit_gap_visit_count_comparison.csv 发来复核。")


if __name__ == "__main__":
    main()