"""Recover contact-inclusive final-model windows and five-frame eligibility."""
from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
from extract_uniform_20frames import candidate_sequences
ROOT=Path(__file__).resolve().parent
FEATURES6=tuple(f'kp_{j}_{a}n' for j in (11,12,13,14,15,16) for a in 'xy')
def windows(samples,condition,include_contact=False):
 count={'pre3':3,'pre5':5}[condition]
 rows=[]
 for name in ['Combined_penalty_pose_keypoints_normalized.csv','150penalties_pose_keypoints_normalized.csv']:
  rows.extend(csv.DictReader((ROOT/'Main Data'/name).open()))
 candidates=defaultdict(list)
 for c in candidate_sequences(rows):candidates[c['key'][:2]].append(c)
 result=[];audit=[]
 for sample in samples:
  selected=sample['rows'];end=float(selected[-1]['frame']);start=float(selected[0]['frame']);assert end.is_integer()
  matches=[]
  for c in candidates[(sample['dataset'],sample['kick_id'])]:
   if c['key'][2]!=selected[0]['camera_direction']:continue
   byframe={float(rows[i]['frame']):i for i in c['indices']}
   if start not in byframe or end not in byframe:continue
   if all(np.allclose([float(rows[byframe[f]][col]) for col in FEATURES6],[float(r[col]) for col in FEATURES6],rtol=0,atol=1e-9) for f,r in [(start,selected[0]),(end,selected[-1])]):matches.append(byframe)
  unique={}
  for bf in matches:
   signature=tuple((f,tuple(rows[i][col] for col in FEATURES6)) for f,i in sorted(bf.items()) if start<=f<=end)
   unique.setdefault(signature,bf)
  matches=list(unique.values())
  assert len(matches)==1,(sample['group_id'],len(matches))
  stop=int(end)+int(include_contact)
  target=list(range(stop-count,stop));bf=matches[0]
  missing=[f for f in target if f not in bf]
  if missing:
   result.append(np.full((count,12),np.nan,dtype=np.float32));audit.append({'group_id':sample['group_id'],'contact_frame':int(end),'frames':target,'missing_frames':missing});continue
  x=np.array([[float(rows[bf[f]][col]) for col in FEATURES6] for f in target],dtype=np.float32)
  assert np.isfinite(x).all() and x.shape==(count,12)
  result.append(x);audit.append({'group_id':sample['group_id'],'contact_frame':int(end),'frames':target,'source_csv_lines_in_concatenated_inputs':[bf[f]+2 for f in target],'contact_status':selected[0].get('contact_status','')})
 return np.stack(result),audit
