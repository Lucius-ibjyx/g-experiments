from __future__ import annotations

from collections import defaultdict
import math
import pandas as pd


def _local_point(x,y,b):
    cx=(b.xtl+b.xbr)/2;cy=(b.ytl+b.ybr)/2
    angle=math.radians(float(getattr(b,'rotation',0)))
    dx=x-cx;dy=y-cy
    return cx+math.cos(angle)*dx+math.sin(angle)*dy,cy-math.sin(angle)*dx+math.cos(angle)*dy

def _inside(x,y,b):
    lx,ly=_local_point(x,y,b)
    return b.xtl<=lx<=b.xbr and b.ytl<=ly<=b.ybr


def _pixel_to_scene_angles(x: float, y: float, width: float, height: float,
                           hfov_deg: float, vfov_deg: float) -> tuple[float, float]:
    """Approximate scene-camera pixel -> angular coordinates using a pinhole model.

    This is used ONLY to express AOI-boundary proximity in degrees. It is independent
    from the head-centered 3D gaze angles used for the main spatial-distribution analysis.
    """
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
    ax = math.degrees(math.atan((x - width / 2.0) / fx))
    ay = math.degrees(math.atan((height / 2.0 - y) / fy))
    return ax, ay


def _distance_to_box_deg(x: float, y: float, b, width: float, height: float,
                         hfov_deg: float, vfov_deg: float) -> float:
    """Angular distance from a gaze pixel to the nearest point of a rectangular AOI.

    Returns 0 for an exact inside hit. For outside points, the closest point on the
    rectangle is projected to approximate scene-camera angular coordinates and the
    Euclidean angular separation is returned.
    """
    lx,ly=_local_point(x,y,b)
    qlx=min(max(lx,float(b.xtl)),float(b.xbr)); qly=min(max(ly,float(b.ytl)),float(b.ybr))
    cx=(b.xtl+b.xbr)/2;cy=(b.ytl+b.ybr)/2
    a=math.radians(float(getattr(b,"rotation",0)))
    qx=cx+math.cos(a)*(qlx-cx)-math.sin(a)*(qly-cy)
    qy=cy+math.sin(a)*(qlx-cx)+math.cos(a)*(qly-cy)
    if _inside(x,y,b):
        return 0.0
    ax1, ay1 = _pixel_to_scene_angles(x, y, width, height, hfov_deg, vfov_deg)
    ax2, ay2 = _pixel_to_scene_angles(qx, qy, width, height, hfov_deg, vfov_deg)
    return math.hypot(ax1 - ax2, ay1 - ay2)


def _resolve_exclusive(candidates):
    """Resolve foreground/background conflicts within candidates of the same hit tier."""
    excl = [c for c in candidates if c[0].relation == "exclusive"]
    if not excl:
        return candidates
    best = max(excl, key=lambda c: (-999 if pd.isna(c[0].priority) else c[0].priority))
    return [c for c in candidates if c[0].relation != "exclusive"] + [best]


def add_aoi_hits(
    gaze: pd.DataFrame,
    boxes: pd.DataFrame,
    *,
    video_width: int,
    video_height: int,
    tolerance_deg: float = 0.0,
    scene_hfov_deg: float = 95.0,
    scene_vfov_deg: float = 63.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match gaze samples to dynamic CVAT rectangles with dual hit status.

    Frozen rules for V0.1 beta:
    - outside=1: object is not visible and is not eligible for any hit.
    - occluded is preserved but NOT used as a blanket exclusion rule.
    - exact_hit: gaze point lies inside the CVAT rectangle.
    - tolerance_hit: gaze point lies outside, but within `tolerance_deg` of the AOI boundary.
      Exact and tolerance hits are always distinguishable in output.
    - exact hits outrank tolerance hits when selecting the primary AOI.
    - exclusive AOIs: within the same hit tier, keep highest priority (C10 phone > computer).
    - parent/child AOIs may both hit.
    - doorplate tracks remain physically separate; semantic aggregation ignores sign_type and
      uses target_status + room_id.
    """
    by_frame = defaultdict(list)
    if tolerance_deg != 0: raise ValueError("V1.1只计算精准命中；aoi_tolerance_deg必须为0")
    for row in boxes.itertuples(index=False):
        if int(row.outside) == 0:
            by_frame[int(row.frame)].append(row)

    cols = [
        "aoi_primary_class", "aoi_primary_semantic_key", "aoi_primary_hit_type",
        "aoi_primary_boundary_distance_deg",
        "aoi_exact_classes", "aoi_tolerance_classes",
        "aoi_all_classes", "aoi_all_semantic_keys", "aoi_track_ids",
        "aoi_tolerance_ambiguous",
        "navigation_sign_hit", "navigation_sign_hit_type",
        "navigation_a520_info_hit", "navigation_a520_info_hit_type",
        "doorplate_target_status", "doorplate_room_ids", "doorplate_hit_type",
    ]
    out = gaze.copy()
    results = {c: [] for c in cols}
    long_hits = []

    for r in out.itertuples():
        frame = int(r.video_frame) if pd.notna(r.video_frame) else None
        x = getattr(r, "gaze_x_px")
        y = getattr(r, "gaze_y_px")
        valid = bool(getattr(r, "gaze_valid")) if pd.notna(getattr(r, "gaze_valid")) else False

        exact = []      # list[(box, 0.0)]
        near = []       # list[(box, distance_deg)]

        if frame is not None and valid and pd.notna(x) and pd.notna(y):
            xf, yf = float(x), float(y)
            for b in by_frame.get(frame, []):
                if _inside(xf, yf, b):
                    exact.append((b, 0.0))
                elif tolerance_deg > 0:
                    dist = _distance_to_box_deg(
                        xf, yf, b, float(video_width), float(video_height),
                        scene_hfov_deg, scene_vfov_deg,
                    )
                    if dist <= tolerance_deg:
                        near.append((b, dist))

        # Relationship handling is done separately inside exact and tolerance tiers.
        exact_eff = _resolve_exclusive(exact)
        near_eff = _resolve_exclusive(near)

        # If an exact semantic hit exists, suppress a tolerance copy of that same semantic key.
        exact_sem = {c[0].semantic_key for c in exact_eff}
        near_eff = [c for c in near_eff if c[0].semantic_key not in exact_sem]

        effective = [(b, d, "exact") for b, d in exact_eff] + [(b, d, "tolerance") for b, d in near_eff]

        for b, dist, hit_type in effective:
            long_hits.append({
                "sample_id": getattr(r, "sample_id"),
                "video_time_s": getattr(r, "video_time_s"),
                "video_frame": frame,
                "track_id": b.track_id,
                "raw_label": b.raw_label,
                "standard_class": b.standard_class,
                "semantic_key": b.semantic_key,
                "relation": b.relation,
                "target_status": b.target_status,
                "room_id": b.room_id,
                "sign_type": b.sign_type,
                "hit_type": hit_type,
                "boundary_distance_deg": float(dist),
            })

        exact_classes = sorted({b.standard_class for b, _ in exact_eff})
        near_classes = sorted({b.standard_class for b, _ in near_eff})
        classes = sorted({b.standard_class for b, _, _ in effective})
        semantics = sorted({b.semantic_key for b, _, _ in effective})
        tracks = sorted({int(b.track_id) for b, _, _ in effective})

        # Primary selection: exact > tolerance. Within a tier: child first, otherwise nearest/stable.
        def choose_primary(cands):
            if not cands:
                return None
            child = [c for c in cands if c[0].relation == "child"]
            pool = child if child else cands
            return sorted(pool, key=lambda c: (float(c[1]), c[0].standard_class, int(c[0].track_id)))[0]

        primary_pair = choose_primary(exact_eff)
        primary_type = "exact" if primary_pair else None
        if primary_pair is None:
            primary_pair = choose_primary(near_eff)
            primary_type = "tolerance" if primary_pair else None

        primary = primary_pair[0] if primary_pair else None
        primary_dist = primary_pair[1] if primary_pair else None

        # Ambiguous near-hit = no exact hit and >1 distinct semantic keys inside tolerance.
        near_semantics = {b.semantic_key for b, _ in near_eff}
        tolerance_ambiguous = int(not exact_eff and len(near_semantics) > 1)

        results["aoi_primary_class"].append(primary.standard_class if primary else pd.NA)
        results["aoi_primary_semantic_key"].append(primary.semantic_key if primary else pd.NA)
        results["aoi_primary_hit_type"].append(primary_type if primary else pd.NA)
        results["aoi_primary_boundary_distance_deg"].append(float(primary_dist) if primary_dist is not None else pd.NA)
        results["aoi_exact_classes"].append("|".join(exact_classes) if exact_classes else "")
        results["aoi_tolerance_classes"].append("|".join(near_classes) if near_classes else "")
        results["aoi_all_classes"].append("|".join(classes) if classes else "")
        results["aoi_all_semantic_keys"].append("|".join(semantics) if semantics else "")
        results["aoi_track_ids"].append("|".join(map(str, tracks)) if tracks else "")
        results["aoi_tolerance_ambiguous"].append(tolerance_ambiguous)

        def class_hit_info(cls):
            e = any(b.standard_class == cls for b, _ in exact_eff)
            n = any(b.standard_class == cls for b, _ in near_eff)
            if e:
                return 1, "exact"
            if n:
                return 1, "tolerance"
            return 0, ""

        nav_hit, nav_type = class_hit_info("navigation_sign")
        info_hit, info_type = class_hit_info("navigation_a520_info")
        results["navigation_sign_hit"].append(nav_hit)
        results["navigation_sign_hit_type"].append(nav_type)
        results["navigation_a520_info_hit"].append(info_hit)
        results["navigation_a520_info_hit_type"].append(info_type)

        door_eff = [(b, d, t) for b, d, t in effective if b.standard_class in {"target_doorplate", "non_target_doorplate"}]
        results["doorplate_target_status"].append(
            "|".join(sorted({b.target_status for b, _, _ in door_eff if b.target_status})) if door_eff else ""
        )
        results["doorplate_room_ids"].append(
            "|".join(sorted({b.room_id for b, _, _ in door_eff if b.room_id})) if door_eff else ""
        )
        door_types = {t for _, _, t in door_eff}
        if "exact" in door_types:
            door_type = "exact"
        elif "tolerance" in door_types:
            door_type = "tolerance"
        else:
            door_type = ""
        results["doorplate_hit_type"].append(door_type)

    for c, v in results.items():
        out[c] = v
    return out, pd.DataFrame(long_hits, columns=["sample_id","video_time_s","video_frame","track_id","raw_label","standard_class","semantic_key","relation","target_status","room_id","sign_type","hit_type","boundary_distance_deg"])
