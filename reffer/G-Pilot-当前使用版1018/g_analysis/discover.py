from __future__ import annotations
from pathlib import Path


def _clean(files):
    return [p for p in files if "__MACOSX" not in p.parts and not p.name.startswith("._") and not p.name.startswith(".~")]


def discover_dataset(root: str | Path) -> dict:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    videos = _clean(list(root.rglob("*.mp4")))
    xmls = _clean(list(root.rglob("*.xml")))
    xlsx = _clean(list(root.rglob("*.xlsx")) + list(root.rglob("*.csv")))
    stage_candidates = [p for p in xlsx if any(x in p.name.lower() for x in ["阶段", "stage", "任务时间", "时间表", "interval"])]
    gazes = _clean(list(root.rglob("GazeData(RelativeRecording)*.csv")))

    if len(videos) != 1:
        raise ValueError(f"Expected exactly 1 MP4 under {root}, found {len(videos)}: {videos}")
    if len(xmls) > 1:
        raise ValueError(f"Expected at most 1 XML under {root}, found {len(xmls)}: {xmls}")
    if len(stage_candidates) != 1:
        raise ValueError(f"Expected exactly 1 stage XLSX under {root}, found {len(stage_candidates)}: {stage_candidates}")
    if not gazes:
        raise ValueError(f"No GazeData(RelativeRecording).csv found under {root}")

    # Critical project rule: if there are multiple ErgoLAB exports, prefer the Glasses3Video segment.
    glasses = [p for p in gazes if "Glasses3Video" in str(p)]
    if len(glasses) == 1:
        gaze = glasses[0]
    elif len(gazes) == 1:
        gaze = gazes[0]
    else:
        raise ValueError(
            "Multiple GazeData exports found and Glasses3Video cannot be uniquely selected:\n" +
            "\n".join(map(str, gazes))
        )

    raw_candidates = list(gaze.parent.glob("RawData(RelativeRecording)*.csv"))
    if len(raw_candidates) != 1:
        raise ValueError(f"Expected RawData beside selected GazeData, found: {raw_candidates}")

    return {
        "video": str(videos[0].resolve()),
        "stage": str(stage_candidates[0].resolve()),
        "cvat": str(xmls[0].resolve()) if xmls else None,
        "gaze": str(gaze.resolve()),
        "raw": str(raw_candidates[0].resolve()),
    }
