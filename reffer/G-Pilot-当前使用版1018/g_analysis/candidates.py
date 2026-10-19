"""G -> A2 exploratory candidates, never automatic best/safe positions."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .reporting import read_groups,density,region,scene_figure,safe

def box_angles(cfg):
    if cfg.get('width_deg') and cfg.get('height_deg'):
        return float(cfg['width_deg']),float(cfg['height_deg'])
    keys=['figma_canvas_width_px','figma_canvas_height_px','display_hfov_deg','display_vfov_deg']
    if not all(cfg.get(k) for k in keys) or not cfg.get('full_canvas_maps_to_full_display',False):return None
    w,h=cfg['width_px'],cfg['height_px'];cw,ch=cfg['figma_canvas_width_px'],cfg['figma_canvas_height_px']
    if not (0<w<=cw and 0<h<=ch):raise ValueError('信息框必须位于配置的完整画布内')
    # Approximate rectilinear projection. Figma full canvas must map to actual full display.
    return (float(np.degrees(2*np.arctan(w/cw*np.tan(np.radians(cfg['display_hfov_deg']/2))))),
            float(np.degrees(2*np.arctan(h/ch*np.tan(np.radians(cfg['display_vfov_deg']/2))))))

def rank_candidates(g,width,height,cfg):
    step=float(cfg['scan_step_deg']);lim=float(cfg['forward_domain_limit_deg']);records=[]
    by=list(g.groupby('participant_id'))
    edges=np.arange(-90.,91.);_,sm=density(g,edges,edges,1.5)
    r50=region(sm,.5,1)[0];r80=region(sm,.8,1)[0]
    for h in np.arange(-lim+width/2,lim-width/2+1e-9,step):
        for v in np.arange(-lim+height/2,lim-height/2+1e-9,step):
            masses=[];supports=[]
            for _,p in by:
                masses.append(float((p.horizontal_visual_angle_deg.between(h-width/2,h+width/2)&p.vertical_visual_angle_deg.between(v-height/2,v+height/2)).mean()))
                supports.append(bool((p.horizontal_visual_angle_deg.between(h-cfg['support_radius_deg'],h+cfg['support_radius_deg'])&p.vertical_visual_angle_deg.between(v-cfg['support_radius_deg'],v+cfg['support_radius_deg'])).any()))
            ecc=float(np.degrees(np.arctan(np.hypot(np.tan(np.radians(h)),np.tan(np.radians(v))))))
            dx=np.maximum(0,np.minimum(edges[1:],h+width/2)-np.maximum(edges[:-1],h-width/2))
            dy=np.maximum(0,np.minimum(edges[1:],v+height/2)-np.maximum(edges[:-1],v-height/2))
            cell_area=dy[:,None]*dx[None,:]
            records.append({'h_deg':h,'v_deg':v,'eccentricity_deg':ecc,'width_deg':width,'height_deg':height,
                'occupancy_mean':np.mean(masses),'occupancy_min':min(masses),'occupancy_max':max(masses),'occupancy_sd':np.std(masses),
                'occupied_area_overlap_50':float((cell_area*r50).sum()/(width*height)),
                'occupied_area_overlap_80':float((cell_area*r80).sum()/(width*height)),
                'participant_n':len(by),'support_participant_fraction':np.mean(supports),
                'eligible':np.mean(supports)>=cfg['min_support_fraction'],
                'note':'观测邻域支持用于筛查，零采样不等于安全；访问成本尚未验证'})
    return pd.DataFrame(records)

def select_candidates(table,cfg):
    eligible=table[table.eligible].copy();selected=[];best=None
    if len(eligible)<2:return selected,'观测支持不足'
    for _,high in eligible.sort_values('occupancy_mean',ascending=False).iterrows():
        options=eligible[(eligible.eccentricity_deg-high.eccentricity_deg).abs()<=cfg['matched_eccentricity_tolerance_deg']]
        options=options[np.hypot(options.h_deg-high.h_deg,options.v_deg-high.v_deg)>=cfg['min_center_separation_deg']]
        if options.empty:continue
        low=options.sort_values('occupancy_mean').iloc[0];diff=high.occupancy_mean-low.occupancy_mean
        if best is None or diff>best[0]:best=(diff,high,low)
    if best is None or best[0]<=0:return selected,'未找到偏心角接近且占用不同的高低候选'
    for role,row in [('high_conflict',best[1]),('low_conflict',best[2])]:
        selected.append({**row.to_dict(),'role':role,'status':'exploratory_not_validated'})
    low=best[2]
    far=eligible[(eligible.occupancy_mean<=low.occupancy_mean+cfg['far_occupancy_margin']) & (eligible.eccentricity_deg>=low.eccentricity_deg+cfg['min_far_eccentricity_difference_deg'])]
    far=far[np.hypot(far.h_deg-low.h_deg,far.v_deg-low.v_deg)>=cfg['min_center_separation_deg']]
    if len(far):
        selected.append({**far.sort_values(['eccentricity_deg','occupancy_mean'],ascending=[False,True]).iloc[0].to_dict(),'role':'possible_high_access_cost','status':'geometric_hypothesis_A2_required'})
        return selected,'三类均为探索性候选；第三类仅为较大偏心角假设'
    return selected,'仅找到高低冲突候选；第三类条件不足，不强行生成'

def generate_candidates(root):
    box=json.loads((root/'config/information_box.json').read_text());cfg=json.loads((root/'config/candidate_settings.json').read_text())
    out=root/'output'/'candidate_positions';out.mkdir(parents=True,exist_ok=True);angles=box_angles(box)
    if angles is None:
        (out/'STATUS.json').write_text(json.dumps({'status':'pending_box_angular_calibration','width_px':box['width_px'],'height_px':box['height_px'],
            'reason':'Figma的422×103px不能直接换算视觉角。填写完整画布、设备H/V视场角及映射关系，或直接填写角宽高。其余分析不受影响。'},ensure_ascii=False,indent=2))
        print('候选评分待信息框角尺寸；其余分析可运行。');return
    width,height=angles
    if min(angles)<=0 or max(angles)>=2*cfg['forward_domain_limit_deg']:raise ValueError('信息框角尺寸超出前向扫描范围')
    groups=read_groups(root);allg=pd.concat([g for _,g in groups],ignore_index=True)
    for task,g in allg.groupby('task_id'):
        td=out/str(task);td.mkdir(exist_ok=True)
        scopes=[('whole_task',g)]+[(str(s),p) for s,p in g.groupby('stage_type') if s!='whole_task']
        for tag,sg in scopes:
            table=rank_candidates(sg,width,height,cfg);table.to_csv(td/f'{safe(tag)}__all_positions.csv',index=False,encoding='utf-8-sig')
            selected,note=select_candidates(table,cfg)
            pd.DataFrame(selected,columns=list(table.columns)+['role','status']).to_csv(td/f'{safe(tag)}__selected.csv',index=False,encoding='utf-8-sig')
            (td/f'{safe(tag)}__selection.json').write_text(json.dumps({'note':note,'box':box,'angles_deg':angles,'rules':cfg,'display_window_restriction':False,
                'domain':'前向投影范围，非完整生理视野','stage':'候选阶段，不自动作为A2最终条件'},ensure_ascii=False,indent=2))
            fig,ax=plt.subplots(figsize=(8,6));p=ax.scatter(table.h_deg,table.v_deg,c=table.occupancy_mean,cmap='magma',s=12)
            unsupported=table[~table.eligible];ax.scatter(unsupported.h_deg,unsupported.v_deg,c='grey',s=7,alpha=.35)
            for r in selected:
                ax.scatter(r['h_deg'],r['v_deg'],s=130,facecolors='none',edgecolors='cyan');ax.annotate(r['role'],(r['h_deg'],r['v_deg']),fontsize=7)
            ax.set(xlabel='Horizontal angle (deg)',ylabel='Vertical angle (deg)',title=f'{task} | {tag}; grey = unsupported')
            fig.colorbar(p,ax=ax,label='Equal-participant sample mass in standard box');fig.tight_layout();fig.savefig(td/f'{safe(tag)}__map.png',dpi=140);plt.close(fig)
            for folder,pg in groups:
                pg=pg[pg.task_id.astype(str)==str(task)]
                if tag!='whole_task':pg=pg[pg.stage_type==tag]
                if pg.empty:continue
                (folder/'report_figures').mkdir(exist_ok=True)
                he=np.arange(-90,91);ve=np.arange(-90,91);_,sm=density(pg,he,ve,1)
                scene_figure(root,folder,pg,sm,he,ve,f'candidate_{safe(tag)}',candidates=selected)
    (out/'STATUS.json').write_text(json.dumps({'status':'exploratory_candidates_generated','angles_deg':angles},ensure_ascii=False,indent=2))
