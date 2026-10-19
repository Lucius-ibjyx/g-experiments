from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def normalize_aoi_label(raw_label: str) -> dict:
    s = raw_label.strip()
    for prefix,cls,priority in [("价格-","price_tag",4),("目标-","item",3),("货架-","shelf",1)]:
        if s.startswith(prefix):
            product=next((p for p in ["饮料","饼干","纸巾"] if p in s),"")
            return {"standard_class":cls,"relation":"exclusive","parent_aoi":None,"priority":priority,"target_status":"target" if cls!="shelf" else None,"room_id":None,"sign_type":None,"semantic_key":s,"product_category":product}
    aliases={"材料":("material",1),"问题":("question",2),"商品区":("shelf",1),"具体商品":("item",3),"商品":("item",3),"价签":("price_tag",4),"价格":("price_tag",4),"区域指示牌":("sign",2),"指示牌":("sign",2),"购物清单":("shopping_list",5),"清单":("shopping_list",5)}
    base=s.split("__",1)[0]
    if base in aliases:
        cls,priority=aliases[base]
        return {"standard_class":cls,"relation":"exclusive","parent_aoi":None,"priority":priority,"target_status":None,"room_id":None,"sign_type":None,"semantic_key":s}

    if s == "显示屏":
        return {"standard_class": "computer_material", "relation": "exclusive", "parent_aoi": None,
                "priority": 1, "target_status": None, "room_id": None, "sign_type": None,
                "semantic_key": "computer_material"}
    if s == "手机屏":
        return {"standard_class": "phone_screen", "relation": "exclusive", "parent_aoi": None,
                "priority": 2, "target_status": None, "room_id": None, "sign_type": None,
                "semantic_key": "phone_screen"}
    if s == "导航牌":
        return {"standard_class": "navigation_sign", "relation": "parent", "parent_aoi": None,
                "priority": None, "target_status": None, "room_id": None, "sign_type": None,
                "semantic_key": "navigation_sign"}
    if "导航牌" in s and "A520" in s:
        return {"standard_class": "navigation_a520_info", "relation": "child", "parent_aoi": "navigation_sign",
                "priority": None, "target_status": "target", "room_id": "A520", "sign_type": None,
                "semantic_key": "navigation_a520_info"}

    m = re.match(r"^(非目标门牌|目标门牌)[-_]?(A\d+)(?:[-_](突出|居中))?$", s)
    if m:
        target_cn, room_id, sign_cn = m.groups()
        target_status = "target" if target_cn == "目标门牌" else "non_target"
        sign_type = {"突出": "protruding", "居中": "centered"}.get(sign_cn)
        standard_class = "target_doorplate" if target_status == "target" else "non_target_doorplate"
        # MAIN analysis semantic ignores protruding/centered by design.
        semantic_key = f"{target_status}__{room_id}"
        return {"standard_class": standard_class, "relation": "independent", "parent_aoi": None,
                "priority": None, "target_status": target_status, "room_id": room_id,
                "sign_type": sign_type, "semantic_key": semantic_key}

    return {"standard_class": "unmapped", "relation": "unknown", "parent_aoi": None,
            "priority": None, "target_status": None, "room_id": None, "sign_type": None,
            "semantic_key": f"unmapped__{s}"}


def parse_cvat_xml(path: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    root = ET.parse(path).getroot()
    if root.findall(".//polygon") or root.findall(".//ellipse"): raise ValueError("本版仅支持矩形/旋转矩形，请勿导入多边形或椭圆")
    original_size = root.find("./meta/original_size")
    if original_size is None:original_size=root.find('.//original_size')
    meta = {
        "width": int(original_size.findtext("width")) if original_size is not None else None,
        "height": int(original_size.findtext("height")) if original_size is not None else None,
        "job_size": int(root.findtext("./meta/job/size")) if root.find("./meta/job/size") is not None else None,
        "start_frame": int(root.findtext("./meta/job/start_frame")) if root.find("./meta/job/start_frame") is not None else None,
        "stop_frame": int(root.findtext("./meta/job/stop_frame")) if root.find("./meta/job/stop_frame") is not None else None,
        "mode": root.findtext("./meta/job/mode"),
    }

    tracks = []
    boxes = []
    for tr in root.findall("track"):
        tid = int(tr.attrib["id"])
        raw = tr.attrib["label"]
        n = normalize_aoi_label(raw)
        tr_boxes = tr.findall("box")
        tracks.append({
            "track_id": tid, "raw_label": raw, **n,
            "first_frame": int(tr_boxes[0].attrib["frame"]) if tr_boxes else None,
            "last_frame": int(tr_boxes[-1].attrib["frame"]) if tr_boxes else None,
        })
        for b in tr_boxes:
            boxes.append({
                "track_id": tid,
                "raw_label": raw,
                **n,
                "frame": int(b.attrib["frame"]),
                "outside": int(b.attrib.get("outside", "0")),
                "occluded": int(b.attrib.get("occluded", "0")),
                "keyframe": int(b.attrib.get("keyframe", "0")),
                "rotation": float(b.attrib.get("rotation", "0")),
                "xtl": float(b.attrib["xtl"]),
                "ytl": float(b.attrib["ytl"]),
                "xbr": float(b.attrib["xbr"]),
                "ybr": float(b.attrib["ybr"]),
            })
    return meta, pd.DataFrame(tracks), pd.DataFrame(boxes)
