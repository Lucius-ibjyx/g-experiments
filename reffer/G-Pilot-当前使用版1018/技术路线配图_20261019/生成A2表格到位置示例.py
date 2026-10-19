from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
plt.rcParams.update({'font.family': 'Arial Unicode MS', 'axes.unicode_minus': False, 'font.size': 13})
source = ROOT / 'output/candidate_positions/T2.1/whole_task__selected.csv'
with source.open(encoding='utf-8-sig', newline='') as f:
    rows = {r['role']: r for r in csv.DictReader(f)}
items = [('A', '较高占用', '#d65c16', rows['high_conflict']),
         ('B', '较低占用', '#16816b', rows['low_conflict'])]
fig = plt.figure(figsize=(15, 8), facecolor='white')
fig.suptitle('A2实例：把视觉占用表中的位置，画到角度坐标中', fontsize=23, y=.96)
fig.text(.5, .895, 'T2.1全任务 · 4位被试 · 同样大小的信息框：12° × 2.67°', ha='center', color='#475569', fontsize=15)
left = fig.add_axes([.035, .23, .43, .59]); left.axis('off')
left.text(0, 1, '① 先读占用表，按数值规则筛选', fontsize=18, va='top', weight='bold')
cells = []
for name, desc, color, row in items:
    cells.append([f'{name}：{desc}', f"{float(row['h_deg']):+.3f}°", f"{float(row['v_deg']):+.3f}°", f"{float(row['occupancy_mean'])*100:.3f}%"])
table = left.table(cellText=cells, colLabels=['信息框', '中心H', '中心V', '视觉占用率'],
                   colWidths=[.29, .23, .23, .25], bbox=[0, .38, 1, .44], cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(12)
for (i,j), cell in table.get_celld().items():
    cell.set_edgecolor('#d4dce5')
    if i == 0: cell.set_facecolor('#edf2f7'); cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor('#fff9f3' if i==1 else '#f0faf6')
        if j in (0,3): cell.set_text_props(color=items[i-1][2], weight='bold')
left.text(0, .25, '每人先算：框内样本数 ÷ 全任务有效空间样本数\n再对4人的比例取平均。', fontsize=14, va='top', linespacing=1.7)
left.text(0, .04, '占用率用于比较；H、V决定框中心画在哪里。', fontsize=13, color='#475569')
ax = fig.add_axes([.55, .23, .41, .59])
ax.set_title('② 按表中坐标绘制信息框', fontsize=18, pad=17, weight='bold')
ax.set_xlim(-20,20); ax.set_ylim(-16,16); ax.set_aspect('equal', adjustable='box')
ax.set_xticks(range(-20,21,10)); ax.set_yticks(range(-15,16,5))
ax.grid(alpha=.17); ax.axhline(0, color='#9ca3af', lw=.9); ax.axvline(0,color='#9ca3af',lw=.9)
ax.set_xlabel('水平角 H（°，右为正）', labelpad=8)
ax.set_ylabel('垂直角 V（°，上为正）', labelpad=6)
for name, desc, color, row in items:
    h,v,w,height = [float(row[k]) for k in ('h_deg','v_deg','width_deg','height_deg')]
    occ=float(row['occupancy_mean'])*100
    ax.add_patch(Rectangle((h-w/2,v-height/2),w,height,edgecolor=color,facecolor=color,alpha=.12,lw=0))
    ax.add_patch(Rectangle((h-w/2,v-height/2),w,height,edgecolor=color,facecolor='none',lw=2.5))
    ax.plot(h,v,'o',color=color,ms=5)
    ax.annotate(f'{name}  ({h:+.3f}°, {v:+.3f}°)\n视觉占用率 {occ:.3f}%',
        xy=(h,v), xytext=(-17,12 if name=='B' else -10.5), fontsize=12, color=color,
        arrowprops={'arrowstyle':'->','color':color,'lw':1.4},
        bbox={'boxstyle':'round,pad=.4','fc':'white','ec':color}, va='center')
arrow=ConnectionPatch(xyA=(1.015,.60),coordsA=left.transAxes,xyB=(-.10,.60),coordsB=ax.transAxes,
    arrowstyle='-|>',mutation_scale=22,lw=2,color='#64748b')
fig.add_artist(arrow)
fig.text(.5,.09,'图中位置来自占用表和数值筛选；此图只展示A、B两个位置，坐标范围为局部放大。',ha='center',fontsize=13,color='#475569')
fig.text(.5,.045,'当前框大小用于流程验证，尚未完成Figma尺寸映射；占用率不直接等同于实际干扰程度。',ha='center',fontsize=12,color='#64748b')
fig.savefig(Path(__file__).with_name('A2_视觉占用表到带坐标信息框.png'),dpi=180,facecolor='white')
plt.close(fig)
