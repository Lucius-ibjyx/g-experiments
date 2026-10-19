from __future__ import annotations
from pathlib import Path
import pandas as pd
import re

def normalize_task_id(value):
    raw=str(value).strip()
    m=re.fullmatch(r'T1\s*任务\s*([12])',raw,re.I)
    if m:return f'T1.{m.group(1)}'
    if re.fullmatch(r'T3\s*任务\s*1',raw,re.I):return 'T3'
    return raw

def parse_timecode(value, timeline_fps=30.0):
    if pd.isna(value): raise ValueError("开始/结束时间不能为空")
    if isinstance(value, (int, float)): return float(value)
    s=str(value).strip()
    if ":" not in s: return float(s)
    parts=s.split(":")
    if len(parts)==4:
        hh,mm,ss,ff=map(int,parts)
        if not (0<=mm<60 and 0<=ss<60 and 0<=ff<timeline_fps): raise ValueError(f"时间码错误: {s}")
        return hh*3600+mm*60+ss+ff/timeline_fps
    if len(parts)==3:
        hh,mm,ss=map(float,parts)
        return hh*3600+mm*60+ss
    raise ValueError(f"请用秒数或 HH:MM:SS:FF: {s}")

def read_and_standardize_stage(path, stage_map, timeline_fps=30.0):
    path=Path(path)
    df=pd.read_csv(path,encoding="utf-8-sig") if path.suffix.lower()==".csv" else pd.read_excel(path)
    df.columns=df.columns.astype(str).str.strip()
    df=df.rename(columns={k:v for k,v in {"T":"任务编号","具体任务名称":"任务名称"}.items() if k in df and v not in df})
    for c in ["开始时间","结束时间"]:
        if c not in df: raise ValueError(f"时间表缺少 {c}")
    df=df.dropna(subset=["开始时间","结束时间"],how="all").reset_index(drop=True)
    def column(c,default=""):
        return df[c].fillna(default).astype(str).str.strip() if c in df else pd.Series(default,index=df.index)
    out=pd.DataFrame(index=df.index)
    out["stage_id"]=column("阶段")
    for i in out.index:
        if not out.loc[i,"stage_id"]: out.loc[i,"stage_id"]=f"S{i+1:03d}"
    if out.stage_id.duplicated().any(): raise ValueError("同一录制中阶段编号必须唯一")
    out["task_id_raw"]=column("任务编号")
    out["task_id"]=out["task_id_raw"].map(normalize_task_id)
    out["task_name_cn"]=column("任务名称")
    out["stage_name_cn"]=column("阶段名称","完整任务").replace("","完整任务")
    out["search_target"]=column("搜索目标")
    out["search_result"]=column("本次结果")
    excluded=out["stage_name_cn"].isin(["等待发题","意外事件","意外","意外处理"])
    out["include_analysis"]=~excluded
    if "纳入分析" in df:
        for i,value in enumerate(column("纳入分析")):
            if value:
                if value.lower() not in {"1","0","是","否","true","false"}: raise ValueError("纳入分析请填写1/0、是/否")
                out.loc[i,"include_analysis"]=(value.lower() in {"1","是","true"}) and not excluded.iloc[i]
    out["exclusion_reason"]=column("排除原因")
    out.loc[~out.include_analysis & out.exclusion_reason.eq(""),"exclusion_reason"]=out.stage_name_cn
    types=[]
    for i,name in out.stage_name_cn.items():
        if name.startswith("寻找") and any(x in name for x in ["饮料","饼干","纸巾"]):
            types.append("product_search")
            if not out.loc[i,"search_target"]: out.loc[i,"search_target"]=next(x for x in ["饮料","饼干","纸巾"] if x in name)
            if not out.loc[i,"search_result"]:
                out.loc[i,"search_result"]="暂未找到" if any(x in name for x in ["失败","未找到","放弃"]) else "成功" if "成功" in name else "未注明"
        elif name in stage_map: types.append(stage_map[name])
        elif not out.loc[i,"include_analysis"]: types.append("excluded_event")
        else: raise ValueError(f"未配置的阶段名称: {name}；请在 config/stage_map.json 中增加映射")
    out["stage_type"]=types
    for src,dst in [("开始动作","start_action"),("结束动作","end_action"),("备注","notes")]: out[dst]=column(src)
    out["start_tc"]=column("开始时间");out["end_tc"]=column("结束时间")
    out["start_s"]=df["开始时间"].map(lambda x:parse_timecode(x,timeline_fps))
    out["end_s"]=df["结束时间"].map(lambda x:parse_timecode(x,timeline_fps))
    out=out.sort_values("start_s").reset_index(drop=True)
    out["duration_s"]=out.end_s-out.start_s
    if (out.duration_s<=0).any() or (out.start_s<0).any(): raise ValueError("时间段必须非负且结束晚于开始")
    if (out.start_s.iloc[1:].to_numpy()<out.end_s.iloc[:-1].to_numpy()).any(): raise ValueError("时间段重叠，请修正")
    out["occurrence"]=out.groupby(["task_id","stage_type"]).cumcount()+1
    return out

def assign_stage(times_s,stages):
    res=pd.DataFrame(index=times_s.index)
    for c in ["stage_id","stage_type","stage_occurrence","interval_task_id","search_target","search_result","include_analysis","exclusion_reason"]:res[c]=pd.Series(pd.NA,index=times_s.index,dtype="object")
    for _,st in stages.iterrows():
        m=(times_s>=st.start_s)&(times_s<st.end_s)
        for dst,src in [("stage_id","stage_id"),("stage_type","stage_type"),("stage_occurrence","occurrence"),("interval_task_id","task_id"),("search_target","search_target"),("search_result","search_result"),("include_analysis","include_analysis"),("exclusion_reason","exclusion_reason")]:res.loc[m,dst]=st[src]
    return res
