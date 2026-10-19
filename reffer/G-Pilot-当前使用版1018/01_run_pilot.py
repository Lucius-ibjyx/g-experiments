from __future__ import annotations
import json, re, shutil, traceback
from pathlib import Path
import pandas as pd
from g_analysis.discover import discover_dataset
from g_analysis.pipeline import run_dataset
from g_analysis.result_layer import build_result_layer
from g_analysis.qc import stage_qc
from g_analysis.naming import safe_id,parse_recording_name,output_name
ROOT=Path(__file__).resolve().parent

def main():
    if not shutil.which("ffprobe"): raise RuntimeError("请先安装 ffmpeg，并确认终端可运行 ffprobe")
    output=ROOT/"output";output.mkdir(exist_ok=True)
    settings=json.loads((ROOT/"config/analysis_settings.json").read_text())
    mapping=json.loads((ROOT/"config/stage_map.json").read_text())
    rows=[]; used={}
    for folder in sorted((ROOT/"data").iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):continue
        try:
            participant,participant_name,recording=parse_recording_name(folder.name)
            found=discover_dataset(folder)
            manifest={**settings,**found,"participant_id":participant,"participant_name":participant_name,"task_id":recording}
            base=output/"_recordings"/folder.name
            print(f"处理录制: {folder.name}",flush=True)
            meta=run_dataset(manifest,mapping,base)
            (base/"manifest_auto.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            fused=pd.read_csv(base/"gaze_stage_aoi_fused.csv")
            stages=pd.read_csv(base/"stages_standardized.csv").fillna("")
            hits=pd.read_csv(base/"aoi_hits_effective_long.csv")
            tracks=pd.read_csv(base/"aoi_tracks_standardized.csv")
            boxes=pd.read_csv(base/"aoi_boxes_standardized.csv")
            for task,st in stages.groupby("task_id",sort=False):
                all_st=st.copy()
                st=st[st.include_analysis==True].copy()
                if st.empty:
                    rows.append({"recording":folder.name,"task_id":task,"status":"EXCLUDED","message":"全部时段排除"})
                    continue
                task=safe_id(task);name=output_name(participant,participant_name,task);identity=f"{participant}__{task}"
                if identity in used: raise ValueError(f"不同录制重复被试/任务编号 {identity}；请使用独立任务编号")
                out=output/name
                marker=out/"pilot_source.json"
                if out.exists() and (not marker.exists() or json.loads(marker.read_text())["recording"]!=folder.name):raise ValueError(f"输出 {name} 已属于其他来源，请改任务编号")
                used[identity]=folder.name;out.mkdir(exist_ok=True)
                g=fused[(fused.task_id.astype(str)==task)&fused.stage_id.isin(st.stage_id)].copy()
                if g.empty: raise ValueError(f"任务 {task} 的时间内没有眼动样本，请检查同步")
                g["aoi_available"]=bool(found["cvat"])
                h=hits[hits.sample_id.isin(g.sample_id)].copy()
                active=boxes[boxes.frame.isin(g.video_frame.dropna().astype(int).unique())] if len(boxes) else boxes
                tr=tracks[tracks.track_id.isin(active.track_id)] if len(active) else tracks.iloc[:0]
                q=stage_qc(st,meta["video"]["duration_s"])
                all_st.to_csv(out/"all_intervals_including_excluded.csv",index=False,encoding="utf-8-sig")
                pd.DataFrame([{"recording":folder.name,"task_id":task,"listed_interval_seconds":all_st.duration_s.sum(),"analysis_seconds":st.duration_s.sum(),"excluded_seconds":all_st.loc[~all_st.include_analysis,"duration_s"].sum(),"note":"不等同任务绩效耗时"}]).to_csv(out/"analysis_time_audit.csv",index=False,encoding="utf-8-sig")
                (out/"source_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
                for df,fn in [(g,"gaze_stage_aoi_fused.csv"),(h,"aoi_hits_effective_long.csv"),(st,"stages_standardized.csv"),(q,"stage_qc.csv"),(tr,"aoi_tracks_standardized.csv"),(active,"aoi_boxes_standardized.csv")]:df.to_csv(out/fn,index=False,encoding="utf-8-sig")
                build_result_layer(g,h,st,q,tr,out,episode_gap_s=float(settings.get("aoi_episode_gap_s",.10)))
                from g_analysis.reporting import aoi_inventory_report
                aoi_inventory_report(g,h,st,tracks,boxes,out)
                # Mutually exclusive primary AOI sample shares; do not sum semantic visit spans.
                class_rows=[]
                for _,stage_row in st.iterrows():
                    vg=g[(g.stage_id==stage_row.stage_id)&(g.gaze_valid==True)]
                    if not found["cvat"]: continue
                    for (cls,tier),cg in vg.dropna(subset=["aoi_primary_class"]).groupby(["aoi_primary_class","aoi_primary_hit_type"]):
                        class_rows.append({"stage_id":stage_row.stage_id,"stage_type":stage_row.stage_type,"search_target":stage_row.get("search_target",""),"search_result":stage_row.get("search_result",""),"aoi_class":cls,"hit_type":tier,"samples":len(cg),"ratio_of_valid_samples":len(cg)/len(vg)})
                pd.DataFrame(class_rows,columns=["stage_id","stage_type","search_target","search_result","aoi_class","hit_type","samples","ratio_of_valid_samples"]).to_csv(out/"primary_aoi_class_by_stage.csv",index=False,encoding="utf-8-sig")
                pd.DataFrame([{"participant_id":participant,"participant_name":participant_name,"task_id":task,"aoi_available":bool(found["cvat"]),"recording":folder.name,"valid_gaze_samples":int(g.gaze_valid.sum()),"samples":len(g)}]).to_csv(out/"task_summary.csv",index=False,encoding="utf-8-sig")
                (out/"pilot_source.json").write_text(json.dumps({"recording":folder.name,"task_id":task,"aoi_available":bool(found["cvat"]),"time_origin":"full video, no task reset"},ensure_ascii=False,indent=2))
                if not found["cvat"]:(out/"NO_AOI.txt").write_text("本任务未标注AOI。AOI表为空，命中比例不可解释为没有看现实对象。主要使用Stage_summary、task_summary及空间图。")
                rows.append({"recording":folder.name,"task_id":task,"status":"OK","message":""})
                print(f"  OK: {name} ({len(g)} samples)",flush=True)
        except Exception as e:
            print(f"ERROR: {folder.name}: {e}",flush=True)
            (output/f"ERROR__{folder.name}.txt").write_text(traceback.format_exc())
            rows.append({"recording":folder.name,"task_id":"","status":"ERROR","message":str(e)})
    pd.DataFrame(rows,columns=["recording","task_id","status","message"]).to_csv(output/"pilot_run_summary.csv",index=False,encoding="utf-8-sig")
    print("完成；检查 output/pilot_run_summary.csv，再运行 03_build_spatial_baseline.py")
    return 1 if any(r["status"]=="ERROR" for r in rows) else 0
if __name__=="__main__":raise SystemExit(main())
