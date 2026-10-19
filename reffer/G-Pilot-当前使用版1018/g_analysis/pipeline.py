from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .io_utils import read_ergo_csv, probe_video, frame_timestamps, save_json
from .stage import read_and_standardize_stage, assign_stage
from .cvat import parse_cvat_xml
from .angles import add_head_centered_angles
from .video_sync import nearest_frame_indices
from .aoi_hit import add_aoi_hits
from .qc import stage_qc, gaze_qc


def _to_bool(s: pd.Series) -> pd.Series:
    """Read exported validity flags; reject unexpected non-empty values."""
    normalized = s.astype('string').str.strip().str.lower()
    values = normalized.map({"true": True, "false": False,
                             "1": True, "0": False, "1.0": True, "0.0": False})
    unknown = s.notna() & normalized.ne('') & values.isna()
    if unknown.any():
        examples = s.loc[unknown].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"有效性字段 {s.name} 出现无法识别的值：{examples}；应为 True/False 或 1/0")
    return values.fillna(False).astype(bool)


def run_dataset(manifest: dict, stage_map: dict, output_dir: str | Path) -> dict:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    video_path = Path(manifest["video"])
    stage_path = Path(manifest["stage"])
    cvat_path = Path(manifest["cvat"]) if manifest.get("cvat") else None
    gaze_path = Path(manifest["gaze"])
    raw_path = Path(manifest["raw"])
    timeline_fps = float(manifest.get("stage_timeline_fps", 30))

    video_meta = probe_video(video_path)
    fts = frame_timestamps(video_path)
    if cvat_path:
        cvat_meta, tracks, boxes = parse_cvat_xml(cvat_path)
        if cvat_meta.get("width")!=video_meta["width"] or cvat_meta.get("height")!=video_meta["height"]:
            raise ValueError(f"XML尺寸={cvat_meta.get('width')}×{cvat_meta.get('height')}；视频尺寸={video_meta['width']}×{video_meta['height']}。请运行02_check_inputs.py核对。未自动缩放AOI，需确认是否为同一视频及仅改变分辨率。")
    else:
        cvat_meta={"aoi_available":False}
        tracks=pd.DataFrame(columns=["track_id","raw_label","standard_class","semantic_key","relation","priority","target_status","room_id","sign_type"])
        boxes=pd.DataFrame(columns=["frame","outside","track_id"] )
    if len(tracks) and (tracks.standard_class=="unmapped").any():
        unknown=tracks.loc[tracks.standard_class=="unmapped","raw_label"].unique().tolist()
        raise ValueError(f"AOI标签未映射，请核对标注名称或配置: {unknown}")
    stages = read_and_standardize_stage(stage_path, stage_map, timeline_fps)
    if stages.empty: raise ValueError("时间表没有有效任务/阶段")
    if stages.end_s.max()>video_meta["duration_s"]+0.1: raise ValueError("时间表超出视频时长")
    stages["task_id"]=stages["task_id"].replace("",manifest["task_id"])

    gaze_meta, gaze = read_ergo_csv(gaze_path)
    raw_meta, raw = read_ergo_csv(raw_path)

    # merge raw+processed gaze by Id/Time. Raw validity and 3D gaze are authoritative for QC/angles.
    use_g = [c for c in ["Id", "Time(s)", "X(px)", "Y(px)", "Eye State Type", "Velocity(degree/s)"] if c in gaze.columns]
    use_r = [c for c in ["Id", "Time(s)", "GazePointValidity", "GazePointX", "GazePointY",
                         "GazePositionValidity", "GazePositionX", "GazePositionY", "GazePositionZ"] if c in raw.columns]
    merged = pd.merge(raw[use_r], gaze[use_g], on=["Id", "Time(s)"], how="outer", validate="one_to_one")
    merged = merged.sort_values("Time(s)").reset_index(drop=True)
    merged = merged.rename(columns={"Id": "sample_id", "Time(s)": "recording_time_s", "X(px)": "gaze_x_px", "Y(px)": "gaze_y_px"})

    gaze_start_s = float(gaze_meta.get("Start Time(s)", raw_meta.get("Start Time(s)", 0)) or 0)
    merged["video_time_s"] = pd.to_numeric(merged["recording_time_s"], errors="coerce") - gaze_start_s
    merged["gaze_valid"] = _to_bool(merged.get("GazePointValidity", pd.Series(False, index=merged.index)))

    # normalized fallback if X(px)/Y(px) absent
    if "gaze_x_px" not in merged.columns:
        merged["gaze_x_px"] = pd.to_numeric(merged["GazePointX"], errors="coerce") * video_meta["width"]
    if "gaze_y_px" not in merged.columns:
        merged["gaze_y_px"] = pd.to_numeric(merged["GazePointY"], errors="coerce") * video_meta["height"]

    merged = add_head_centered_angles(merged)
    merged["spatial_valid"]=_to_bool(merged["GazePositionValidity"]) & np.isfinite(merged[["GazePositionX","GazePositionY","GazePositionZ"]].apply(pd.to_numeric,errors="coerce")).all(axis=1) & (pd.to_numeric(merged["GazePositionZ"],errors="coerce")>0)
    merged.loc[~merged.spatial_valid,["horizontal_visual_angle_deg","vertical_visual_angle_deg","eccentricity_deg"]]=np.nan
    merged["gaze_valid"]=merged.gaze_valid & np.isfinite(merged[["gaze_x_px","gaze_y_px"]].apply(pd.to_numeric,errors="coerce")).all(axis=1)
    merged["video_frame"] = nearest_frame_indices(merged["video_time_s"], fts)
    st_assigned = assign_stage(merged["video_time_s"], stages)
    for c in st_assigned.columns:
        merged[c] = st_assigned[c]

    aoi_tolerance_deg = float(manifest.get("aoi_tolerance_deg", 0.0))
    scene_hfov_deg = float(manifest.get("scene_hfov_deg", 95.0))
    scene_vfov_deg = float(manifest.get("scene_vfov_deg", 63.0))
    fused, long_hits = add_aoi_hits(
        merged, boxes,
        video_width=video_meta["width"],
        video_height=video_meta["height"],
        tolerance_deg=aoi_tolerance_deg,
        scene_hfov_deg=scene_hfov_deg,
        scene_vfov_deg=scene_vfov_deg,
    )
    fused.insert(0, "participant_id", manifest["participant_id"])
    fused.insert(1, "task_id", fused["interval_task_id"])
    fused["participant_name"]=manifest.get("participant_name","")

    # Main semantic doorplate aggregation is intentionally independent of protruding/centered sign_type.
    # sign_type remains in long hit output only as optional traceability.

    tracks.to_csv(outdir / "aoi_tracks_standardized.csv", index=False, encoding="utf-8-sig")
    boxes.to_csv(outdir / "aoi_boxes_standardized.csv", index=False, encoding="utf-8-sig")
    stages.to_csv(outdir / "stages_standardized.csv", index=False, encoding="utf-8-sig")
    fused.to_csv(outdir / "gaze_stage_aoi_fused.csv", index=False, encoding="utf-8-sig")
    long_hits.to_csv(outdir / "aoi_hits_effective_long.csv", index=False, encoding="utf-8-sig")

    sqc = stage_qc(stages, float(fts[-1]))
    sqc.to_csv(outdir / "stage_qc.csv", index=False, encoding="utf-8-sig")
    gqc = gaze_qc(fused)

    metadata = {
        "participant_id": manifest["participant_id"],
        "participant_name":manifest.get("participant_name",""),
        "task_id": manifest["task_id"],
        "stage_time_format": "HH:MM:SS:FF",
        "stage_timeline_fps": timeline_fps,
        "gaze_segment_start_s": gaze_start_s,
        "gaze_frequency_hz": float(gaze_meta.get("Frequency(Hz)", 0) or 0),
        "gaze_segment_name": gaze_meta.get("Segment Name"),
        "recording_name": gaze_meta.get("Recording Name"),
        "subject_name_raw": gaze_meta.get("Subject Name"),
        "video": video_meta,
        "video_frame_timestamp_last_relative_s": fts[-1],
        "cvat": cvat_meta,
        "aoi_track_count": int(len(tracks)),
        "unmapped_aoi_labels": tracks.loc[tracks["standard_class"] == "unmapped", "raw_label"].drop_duplicates().tolist(),
        "gaze_qc": gqc,
        "alignment_rule": "video_time_s = recording_time_s - ErgoLAB Segment Start Time(s); map to nearest relative decoded video frame timestamp",
        "occluded_rule": "preserve but do not exclude hits; use semantic overlap rules instead",
        "outside_rule": "outside=1 is excluded from AOI hit testing",
        "doorplate_grouping_rule": "main analysis groups by target_status + room_id; sign_type is optional trace field only",
        "aoi_tolerance_deg": aoi_tolerance_deg,
        "scene_camera_hfov_deg": scene_hfov_deg,
        "scene_camera_vfov_deg": scene_vfov_deg,
        "aoi_tolerance_rule": "exact only; rectangle boundary inclusive; no angular expansion",
    }
    save_json(metadata, outdir / "metadata.json")
    pd.DataFrame([{k: v for k, v in metadata.items() if not isinstance(v, (dict, list))}]).to_csv(outdir / "metadata_flat.csv", index=False, encoding="utf-8-sig")

    return metadata
