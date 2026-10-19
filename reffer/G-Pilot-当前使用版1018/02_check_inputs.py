"""Read-only input diagnostics. Does not resize AOIs or modify source files."""
from pathlib import Path
import json
import pandas as pd
from g_analysis.discover import discover_dataset
from g_analysis.io_utils import probe_video
from g_analysis.cvat import parse_cvat_xml
from g_analysis.stage import read_and_standardize_stage

ROOT=Path(__file__).resolve().parent

def main():
    settings=json.loads((ROOT/'config/analysis_settings.json').read_text())
    mapping=json.loads((ROOT/'config/stage_map.json').read_text());rows=[];messages=[]
    for folder in sorted((ROOT/'data').iterdir()):
        if not folder.is_dir() or folder.name.startswith('.'):continue
        row={'录制':folder.name}
        try:
            found=discover_dataset(folder);video=probe_video(found['video'])
            row.update(视频宽=video['width'],视频高=video['height'],视频帧数=video['frame_count'],视频秒数=video['duration_s'])
            notes=[]
            if found['cvat']:
                meta,tracks,boxes=parse_cvat_xml(found['cvat'])
                row.update(XML宽=meta['width'],XML高=meta['height'],XML声明帧数=meta['job_size'],实际轨迹数=len(tracks),旋转框数=int(boxes.rotation.ne(0).sum()) if len(boxes) else 0)
                match=meta['width']==video['width'] and meta['height']==video['height'];row['尺寸匹配']=match
                if not match:notes.append('XML和视频尺寸不同；需核对同一视频/缩放/裁切/导出元数据，未自动修改')
                elif not len(tracks):notes.append('有XML但无轨迹，请核对是否CVAT视频追踪导出')
            else:row['尺寸匹配']='无AOI'
            try:
                stages=read_and_standardize_stage(found['stage'],mapping,settings.get('stage_timeline_fps',30))
                row['任务编号']=' | '.join(stages.task_id.unique());row['阶段名称']=' | '.join(stages.stage_name_cn.unique())
            except Exception as e:notes.append(str(e))
            row['检查说明']='；'.join(notes) or '输入检查通过；仍需人工抽查同步'
        except Exception as e:row['检查说明']=str(e)
        rows.append(row);line=f"{folder.name}: "+'；'.join(f'{k}={v}' for k,v in row.items() if k!='录制')
        messages.append(line);print(line,flush=True)
    out=ROOT/'output';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'input_check.csv',index=False,encoding='utf-8-sig')
    (out/'input_check.txt').write_text('\n'.join(messages),encoding='utf-8')
    print('检查完成：output/input_check.csv、output/input_check.txt；原始数据未修改。')

if __name__=='__main__':main()
