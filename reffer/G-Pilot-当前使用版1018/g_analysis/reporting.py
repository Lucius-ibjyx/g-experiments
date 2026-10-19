"""Pilot reporting: sample occupancy, not fixation-event density or peripheral visibility."""
from pathlib import Path
import json,re,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_fonts={f.name for f in font_manager.fontManager.ttflist}
for _font in ['PingFang SC','Heiti SC','Microsoft YaHei','SimHei','Noto Sans CJK SC','Arial Unicode MS']:
    if _font in _fonts:
        plt.rcParams['font.sans-serif']=[_font,'DejaVu Sans'];break
plt.rcParams['axes.unicode_minus']=False
from matplotlib.colors import PowerNorm
from PIL import Image

H='horizontal_visual_angle_deg'
V='vertical_visual_angle_deg'

def truth(s):
    return s.astype(str).str.lower().isin(['true','1'])

def safe(s):
    return re.sub(r'[^\w-]+','_',str(s))

def polar_xy(h,v):
    """Gnomonic angular coordinates -> forward-centred azimuthal equidistant plane.
    +horizontal right, +vertical up. Radius is true 3D eccentricity in degrees.
    This is a projection conversion, NOT a camera-to-eye calibration.
    """
    x=np.tan(np.radians(np.asarray(h,dtype=float)))
    y=np.tan(np.radians(np.asarray(v,dtype=float)))
    r=np.hypot(x,y)
    ecc=np.degrees(np.arctan(r))
    scale=np.divide(ecc,r,out=np.ones_like(r),where=r>1e-12)
    return x*scale,y*scale

def smooth(a,sigma):
    if sigma<=0:return a.copy()
    n=int(np.ceil(3*sigma)); x=np.arange(-n,n+1)
    k=np.exp(-x*x/(2*sigma*sigma));k/=k.sum()
    def conv(v):return np.convolve(np.pad(v,n),k,mode='valid')
    return np.apply_along_axis(conv,1,np.apply_along_axis(conv,0,a))

def polar_raster(sm,he,ve):
    # Inverse equidistant projection, bilinear render only; analysis stays on H/V grid.
    axis=np.linspace(-90,90,901);px,py=np.meshgrid(axis,axis);r=np.hypot(px,py)
    scale=np.divide(np.tan(np.radians(np.minimum(r,89.999))),r,out=np.ones_like(r),where=r>0)
    hh=np.degrees(np.arctan(px*scale));vv=np.degrees(np.arctan(py*scale))
    ix=(hh-(he[0]+he[1])/2)/(he[1]-he[0]);iy=(vv-(ve[0]+ve[1])/2)/(ve[1]-ve[0])
    inside=(r<90)&(ix>=0)&(iy>=0)&(ix<sm.shape[1]-1)&(iy<sm.shape[0]-1)
    x0=np.clip(np.floor(ix).astype(int),0,sm.shape[1]-2);y0=np.clip(np.floor(iy).astype(int),0,sm.shape[0]-2)
    dx=np.clip(ix-x0,0,1);dy=np.clip(iy-y0,0,1)
    a=sm[y0,x0]*(1-dx)*(1-dy)+sm[y0,x0+1]*dx*(1-dy)+sm[y0+1,x0]*(1-dx)*dy+sm[y0+1,x0+1]*dx*dy
    return np.ma.masked_where(~inside|(a<=sm.max()*.02),a)

def density(g,he,ve,sigma):
    # Equal participant contribution; uniform exported resampling -> sample time mass.
    counts=g.groupby('participant_id')[H].transform('size')
    w=1/counts/g.participant_id.nunique()
    raw=np.histogram2d(g[H],g[V],bins=(he,ve),weights=w)[0].T
    sm=smooth(raw,sigma)
    # Preserve total finite-grid mass after boundary smoothing.
    if sm.sum()>0:sm*=raw.sum()/sm.sum()
    return raw,sm

def region(sm,fraction,bin_size):
    flat=np.sort(sm.ravel())[::-1]
    if flat.sum()<=0:return np.zeros_like(sm,dtype=bool),np.nan,0.
    k=min(int(np.searchsorted(np.cumsum(flat),fraction*flat.sum())),len(flat)-1)
    threshold=flat[k]
    mask=(sm>=threshold)&(sm>0)
    return mask,float(threshold),float(mask.sum()*bin_size**2)

def aoi_inventory_report(g,h,st,tracks,boxes,out):
    rows=[]
    for _,s in st.iterrows():
        sg=g[g.stage_id==s.stage_id]; valid=sg[truth(sg.gaze_valid)]
        for label in tracks.raw_label.drop_duplicates():
            tr=tracks[tracks.raw_label==label]
            bb=boxes[(boxes.raw_label==label)&boxes.frame.isin(sg.video_frame)&(boxes.outside==0)] if len(boxes) else boxes
            ids=valid.loc[valid.video_frame.isin(bb.frame),'sample_id'] if len(bb) else pd.Series(dtype=float)
            hits=h[(h.raw_label==label)&h.sample_id.isin(ids)] if len(h) else h
            rows.append({'stage_id':s.stage_id,'raw_label':label,'standard_class':tr.standard_class.iloc[0],
                         'visible_annotated_frames':bb.frame.nunique() if len(bb) else 0,
                         'valid_samples_during_visible_annotation':len(ids),
                         'effective_hit_samples':hits.sample_id.nunique() if len(ids) else np.nan,
                         'ratio_during_visible_annotation':hits.sample_id.nunique()/len(ids) if len(ids) else np.nan,
                         'status':'visible_annotation' if len(ids) else 'no_visible_annotation_or_no_valid_samples',
                         'note':'仅解释编码范围；没有框无法区分不可见和未编码；不是全阶段对象访问率'})
    pd.DataFrame(rows,columns=['stage_id','raw_label','standard_class','visible_annotated_frames','valid_samples_during_visible_annotation','effective_hit_samples','ratio_during_visible_annotation','status','note']).to_csv(out/'aoi_annotation_scope.csv',index=False,encoding='utf-8-sig')
    (out/'AOI解释.txt').write_text('主AOI优先级：清单 > 价格 > 目标商品 > 货架。未命中仅称已标AOI外。\n'+
        '本版不推断未标注对象的访问率；具体标注窗口见aoi_annotation_scope.csv。\n'+
        '价格-目标纸巾没有实际框时不创建零访问结果。旧AOI_summary只针对实际框，跨全程解释需结合编码范围。\n'+
        'AOI访问持续区间包含gap合并的间隔，不等同有效命中时长；样本命中与访问跨度分别解释。',encoding='utf-8')

def read_groups(root):
    from .run_quality import eligible_datasets
    eligible = eligible_datasets(root/'output')
    groups=[]
    for folder in sorted((root/'output').glob('*')):
        p=folder/'gaze_stage_aoi_fused.csv'
        if not p.exists() or folder.name.startswith('_') or folder.name not in eligible:continue
        g=pd.read_csv(p); flag='spatial_valid' if 'spatial_valid' in g else 'gaze_valid'
        ok=truth(g[flag]) & np.isfinite(g[[H,V]].apply(pd.to_numeric,errors='coerce')).all(axis=1)
        if 'include_analysis' in g:ok &= truth(g.include_analysis)
        g=g[ok].copy()
        if len(g):groups.append((folder,g))
    return groups

def add_contours(ax,sm,he,ve):
    x=(he[:-1]+he[1:])/2;y=(ve[:-1]+ve[1:])/2
    for f,color in [(.8,'#242424'),(.5,'#FFFFFF')]:
        _,threshold,_=region(sm,f,he[1]-he[0])
        if np.isfinite(threshold) and sm.max()>threshold>sm.min():
            ax.contour(x,y,sm,levels=[threshold],colors=[color],linewidths=1)

def background(ax,root):
    cfg=json.loads((root/'config/report_settings.json').read_text())
    ref=cfg['visual_field_reference']; p=root/ref['file']
    if p.exists():
        im=np.asarray(Image.open(p)); height,width=im.shape[:2]
        cx,cy=ref['center_px']; scale=ref['outer_radius_deg']/ref['outer_radius_px']
        ax.imshow(im,extent=[-cx*scale,(width-cx)*scale,-(height-cy)*scale,cy*scale],zorder=0)
    else:
        ax.text(0,85,'Reference image not configured',ha='center',fontsize=8)
    ax.set_xlim(-103,103);ax.set_ylim(-100,103);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])

def scene_figure(root,folder,g,sm,he,ve,tag,candidates=None,heat_vmax=None):
    manifest=json.loads((folder/'source_manifest.json').read_text())
    t=float(g.video_time_s.median()); snapshot=folder/'report_figures'/f'{tag}__context.jpg'
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',manifest['video'],'-frames:v','1',str(snapshot)],check=True)
    im=np.asarray(Image.open(snapshot)); ht,wd=im.shape[:2]
    hf=float(manifest.get('scene_hfov_deg',95)); vf=float(manifest.get('scene_vfov_deg',63))
    # Pixel boundaries through an approximate pinhole camera; do not stretch linearly in degrees.
    xs=np.degrees(np.arctan((np.arange(wd+1)-wd/2)/(wd/2)*np.tan(np.radians(hf/2))))
    ys=np.degrees(np.arctan((ht/2-np.arange(ht+1))/(ht/2)*np.tan(np.radians(vf/2))))
    fig,ax=plt.subplots(figsize=(9,5.6))
    # Decimate RGB image for a tractable angular mesh while preserving camera projection.
    step=6; xx=np.r_[np.arange(0,wd,step),wd];yy=np.r_[np.arange(0,ht,step),ht]
    rgb=im[yy[:-1,None],xx[None,:-1],:]/255
    ax.pcolormesh(xs[xx],ys[yy],rgb,shading='flat',rasterized=True)
    if candidates is None:
        masked=np.ma.masked_where(sm<=sm.max()*.03,sm)
        ax.pcolormesh(he,ve,masked,cmap='magma',alpha=.48,shading='flat',norm=PowerNorm(.5,vmin=0,vmax=heat_vmax))
        add_contours(ax,sm,he,ve)
    else:
        from matplotlib.patches import Rectangle
        for r in candidates:
            ax.add_patch(Rectangle((r['h_deg']-r['width_deg']/2,r['v_deg']-r['height_deg']/2),r['width_deg'],r['height_deg'],fill=False,edgecolor='yellow',linewidth=2))
            ax.text(r['h_deg'],r['v_deg'],r['role'],fontsize=8,color='yellow')
    ax.set(xlim=(-hf/2,hf/2),ylim=(-vf/2,vf/2),xlabel='Horizontal angle (deg; right +)',ylabel='Vertical angle (deg; up +)',
           title=f'{tag} | representative background at {t:.2f}s')
    fig.text(.5,.015,'Context only; not registered object heat. Camera model approximate. White=50%, black=80% smoothed mass.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(snapshot.with_suffix('.png'),dpi=140);plt.close(fig)
    return t

def generate_reports(root):
    cfg=json.loads((root/'config/report_settings.json').read_text());groups=read_groups(root)
    if not groups:raise ValueError('请先运行01；没有可分析空间样本')
    allg=pd.concat([g for _,g in groups],ignore_index=True)
    b=float(cfg['bin_deg']);sigma=float(cfg['smoothing_sigma_deg'])/b
    hl=np.ceil((allg[H].abs().max()+5)/5)*5;vl=np.ceil((allg[V].abs().max()+5)/5)*5
    he=np.arange(-hl,hl+b,b);ve=np.arange(-vl,vl+b,b)
    panels=[]
    for folder,g in groups:
        panels.append((folder,g,'whole_task',False))
        for st,sg in g.groupby('stage_type'):
            if st!='whole_task':panels.append((folder,sg,str(st),False))
    for task,tg in allg.groupby('task_id'):
        out=root/'output'/'group_reports'/str(task);out.mkdir(parents=True,exist_ok=True)
        panels.append((out,tg,'whole_task',True))
        for st,sg in tg.groupby('stage_type'):
            if st!='whole_task':panels.append((out,sg,str(st),True))
    prepared=[(*p,*density(p[1],he,ve,sigma)) for p in panels]
    vmax=max(x[-1].max() for x in prepared);rows=[]
    for folder,g,tag,isgroup,raw,sm in prepared:
        fd=folder/'report_figures';fd.mkdir(exist_ok=True)
        total=sm.sum()
        for f in [.5,.8]:
            mask,threshold,area=region(sm,f,b)
            # Do not silently call a smoothed mass contour an exact raw-sample percentile.
            rows.append({'dataset':str(folder.relative_to(root/'output')),'scope':tag,'participant_n':g.participant_id.nunique(),
                         'aoi_participant_n':g.loc[truth(g.aoi_available),'participant_id'].nunique() if 'aoi_available' in g else 0,
                         'samples':len(g),'target_mass':f,'smoothed_mass_covered':sm[mask].sum()/total,
                         'raw_sample_mass_covered':raw[mask].sum()/raw.sum(),'projected_grid_area_deg2':area,
                         'h_min_deg':he[np.where(mask)[1].min()] if mask.any() else np.nan,
                         'h_max_deg':he[np.where(mask)[1].max()+1] if mask.any() else np.nan,
                         'v_min_deg':ve[np.where(mask)[0].min()] if mask.any() else np.nan,
                         'v_max_deg':ve[np.where(mask)[0].max()+1] if mask.any() else np.nan})
            np.savez_compressed(fd/f'{safe(tag)}__region_{int(f*100)}.npz',mask=mask,h_edges=he,v_edges=ve)
        for zoom in [False,True]:
            fig,ax=plt.subplots(figsize=(7.5,5.8));im=ax.pcolormesh(he,ve,sm,cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=vmax),shading='flat')
            add_contours(ax,sm,he,ve);ax.axhline(0,lw=.5,c='grey');ax.axvline(0,lw=.5,c='grey')
            if zoom:
                # One global P0.5--P99.5 symmetric range for all panels; report raw coverage.
                zh=np.ceil(max(abs(allg[H].quantile(.005)),abs(allg[H].quantile(.995)))/5)*5
                zv=np.ceil(max(abs(allg[V].quantile(.005)),abs(allg[V].quantile(.995)))/5)*5
                inside=g[H].between(-zh,zh)&g[V].between(-zv,zv)
                counts=g.groupby('participant_id')[H].transform('size')
                coverage=float(((inside/counts).groupby(g.participant_id).sum()).mean())
                ax.set_xlim(-zh,zh);ax.set_ylim(-zv,zv)
                ax.text(.02,.02,f'Raw mass inside: {coverage:.2%}; outside: {1-coverage:.2%}',transform=ax.transAxes,color='white',fontsize=8)
            ax.set(xlabel='Horizontal angle (deg; right +)',ylabel='Vertical angle (deg; up +)',title=f'{folder.name} | {tag} | n={g.participant_id.nunique()}')
            fig.colorbar(im,ax=ax,label=f'Equal-participant mass / {b:g} deg grid cell')
            fig.tight_layout();fig.savefig(fd/f'{safe(tag)}__{"zoom" if zoom else "full"}.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7.5,7.5));background(ax,root)
        ax.imshow(polar_raster(sm,he,ve),origin='lower',extent=[-90,90,-90,90],cmap='YlOrRd',norm=PowerNorm(.5,vmin=0,vmax=vmax),alpha=.7,zorder=2,interpolation='bilinear')
        ax.set_title(f'{folder.name} | {tag}: forward-reference projection')
        fig.text(.5,.01,'Literature field: scale reference only. Device forward axis approximately aligned; not retinal/peripheral classification.',ha='center',fontsize=7)
        fig.tight_layout(rect=[0,.03,1,1]);fig.savefig(fd/f'{safe(tag)}__visual_field.png',dpi=140);plt.close(fig)
        # Standalone transparent continuous cloud for presentation composition.
        fig,ax=plt.subplots(figsize=(7.5,5.8));ax.imshow(np.ma.masked_where(sm<=sm.max()*.02,sm),origin='lower',extent=[he[0],he[-1],ve[0],ve[-1]],cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=vmax),interpolation='bilinear');ax.axis('off');fig.savefig(fd/f'{safe(tag)}__cloud_transparent.png',transparent=True,dpi=140,bbox_inches='tight',pad_inches=0);plt.close(fig)
        if not isgroup:scene_figure(root,folder,g,sm,he,ve,safe(tag),heat_vmax=vmax)
    out=root/'output'/'group_reports';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'coverage_regions.csv',index=False,encoding='utf-8-sig')
    (out/'report_parameters.json').write_text(json.dumps({**cfg,'full_extent':[he[0],he[-1],ve[0],ve[-1]],'common_color_vmax':float(vmax),'color_normalization':'PowerNorm gamma=0.5, common vmax','group_rule':'equal participant; no pooling across tasks','metric':'valid spatial samples, uniformly resampled; not fixation events','region_method':'highest smoothed grid mass with tied cells; actual raw mass reported separately'},ensure_ascii=False,indent=2))
    return groups,he,ve
