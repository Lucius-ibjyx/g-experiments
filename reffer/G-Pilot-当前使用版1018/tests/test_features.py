import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from g_analysis.stage import read_and_standardize_stage,assign_stage,parse_timecode
from g_analysis.cvat import parse_cvat_xml,normalize_aoi_label
from g_analysis.aoi_hit import add_aoi_hits
from g_analysis.reporting import polar_xy,density,region
from g_analysis.candidates import box_angles,select_candidates
from g_analysis.naming import parse_recording_name,output_name
MAP=json.loads((ROOT/'config/stage_map.json').read_text())
class Features(unittest.TestCase):
 def test_named_recordings(self):
  self.assertEqual(parse_recording_name('P01_黄妍_T4'),('P01','黄妍','T4'))
  self.assertEqual(parse_recording_name('P01_黄妍_T1.1+T1.2+T3'),('P01','黄妍','T1.1+T1.2+T3'))
  self.assertEqual(output_name('P01','黄妍','T1.1'),'P01_黄妍_T1.1')
  self.assertEqual(parse_recording_name('hy__T4'),('hy','','T4'))
  with self.assertRaises(ValueError):parse_recording_name('P01_黄妍_T1.1/1.2')
 def test_30fps_and_three_tasks(self):
  self.assertAlmostEqual(parse_timecode('00:00:01:28'),1+28/30)
  t=read_and_standardize_stage(ROOT/'templates/任务时间表_三任务合录_示例.csv',MAP)
  a=assign_stage(pd.Series([10,130,150,240,260,419.9,420]),t)
  self.assertEqual(a.interval_task_id.iloc[0],'T1.1');self.assertTrue(pd.isna(a.interval_task_id.iloc[1]));self.assertEqual(a.interval_task_id.iloc[2],'T1.2')
  self.assertTrue(pd.isna(a.interval_task_id.iloc[-1]))
 def test_excluded_and_preparation(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'stage.csv'
   pd.DataFrame({'阶段名称':['领取购物筐','等待发题','意外事件','移动至起点'],'开始时间':[0,1,2,3],'结束时间':[1,2,3,4]}).to_csv(p,index=False)
   s=read_and_standardize_stage(p,MAP)
   self.assertEqual(s.include_analysis.tolist(),[True,False,False,True])
   self.assertEqual(assign_stage(pd.Series([1.5,3.5]),s).include_analysis.tolist(),[False,True])
 def test_exact_priority_outside_and_multiple_tracks(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'a.xml';p.write_text('<annotations><meta><original_size><width>100</width><height>100</height></original_size></meta>'+''.join(f'<track id="{i}" label="{label}"><box frame="0" outside="0" xtl="0" ytl="0" xbr="10" ybr="10"/></track>' for i,label in enumerate(['货架-饮料','目标-饮料','价格-目标饮料','清单']))+'</annotations>')
   _,tr,b=parse_cvat_xml(p)
   g=pd.DataFrame({'sample_id':[1,2],'video_frame':[0,0],'video_time_s':[0,.014],'gaze_x_px':[5,10.001],'gaze_y_px':[5,5],'gaze_valid':[True,True]})
   f,h=add_aoi_hits(g,b,video_width=100,video_height=100)
   self.assertEqual(f.aoi_primary_class.iloc[0],'shopping_list');self.assertTrue(pd.isna(f.aoi_primary_class.iloc[1]));self.assertEqual(set(h.hit_type),{'exact'})
   for priority in ['shopping_list','price_tag','item','shelf']:
    f,h=add_aoi_hits(g,b,video_width=100,video_height=100);self.assertEqual(f.aoi_primary_class.iloc[0],priority)
    b=b[b.standard_class!=priority]
   b.loc[:,'outside']=1
   self.assertTrue(add_aoi_hits(g,b,video_width=100,video_height=100)[1].empty)
 def test_projection_geometry(self):
  x,y=polar_xy([0,30,0,30],[0,0,-30,30]);self.assertAlmostEqual(x[1],30);self.assertAlmostEqual(y[2],-30)
  self.assertAlmostEqual(np.hypot(x[3],y[3]),np.degrees(np.arctan(np.sqrt(2)*np.tan(np.radians(30)))))
 def test_equal_participant_and_mass(self):
  g=pd.DataFrame({'participant_id':['a']*100+['b'],'horizontal_visual_angle_deg':[0]*100+[2],'vertical_visual_angle_deg':[0]*101})
  raw,sm=density(g,np.arange(-5,6),np.arange(-5,6),1)
  self.assertAlmostEqual(raw.sum(),1);self.assertAlmostEqual(raw[:,5].sum(),.5);self.assertAlmostEqual(sm.sum(),1)
  mask,_,_=region(sm,.8,1);self.assertGreaterEqual(sm[mask].sum(),.8-1e-12)
 def test_box_pending_and_no_forced_third(self):
  self.assertIsNone(box_angles({'width_px':422,'height_px':103}))
  self.assertEqual(box_angles({'width_deg':10,'height_deg':3}),(10.,3.))
  cfg=json.loads((ROOT/'config/candidate_settings.json').read_text())
  rows=[{'eligible':True,'h_deg':-10,'v_deg':0,'eccentricity_deg':10,'occupancy_mean':.5},{'eligible':True,'h_deg':10,'v_deg':0,'eccentricity_deg':10,'occupancy_mean':.1}]
  chosen,note=select_candidates(pd.DataFrame(rows),cfg)
  self.assertEqual(len(chosen),2);self.assertNotIn('possible_high_access_cost',[c['role'] for c in chosen])
if __name__=='__main__':unittest.main()
