"""Eight-joint descriptive curves for correctly classified central-camera test clips."""
from pathlib import Path
import csv,json,hashlib
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/experiment3_central_contact_correct_lr_8joint_curves'
JOINTS=[5,6,11,12,13,14,15,16]
NAMES=['Left shoulder','Right shoulder','Left hip','Right hip','Left knee','Right knee','Left ankle','Right ankle']
EDGES=[(0,1),(0,2),(1,3),(2,3),(2,4),(4,6),(3,5),(5,7)]
COLORS={'L':'#2374b5','R':'#e47b23'}
def main():
 OUT.mkdir(exist_ok=False)
 run=ROOT/'runs/experiment3_central_contact_pre3_seed42';data=ROOT/'data/experiment3_central_contact_pre3_seed42'
 paths=[run/'test_integrated_predictions.csv',data/'test_selected_windows.csv',ROOT/'Main Data/Combined_penalty_pose_keypoints_normalized.csv',ROOT/'Main Data/150penalties_pose_keypoints_normalized.csv']
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 predictions=list(csv.DictReader(paths[0].open()));correct=[r for r in predictions if r['actual']==r['predicted']]
 selected=defaultdict(list)
 for r in csv.DictReader(paths[1].open()):selected[(r['dataset'],r['kick_id'])].append(r)
 sources=defaultdict(list)
 for p in paths[2:]:
  for r in csv.DictReader(p.open()):
   try: frame=float(r['frame'])
   except ValueError: continue
   sources[(r['dataset'],r['kick_id'],frame)].append(r)
 groups={'L':[],'R':[]};membership=[]
 features=[f'kp_{j}_{a}n' for j in JOINTS for a in 'xy']
 for pred in correct:
  key=(pred['dataset'],pred['kick_id']);seq=selected[key];assert len(seq)==3
  poses=[]
  for s in seq:
   frame=float(s['source_frame']);candidates=[r for r in sources[(*key,frame)] if r['camera_direction']=='C' and all(np.isclose(float(r[c]),float(s[c]),atol=1e-6,rtol=0) for c in s if c.startswith('kp_'))]
   distinct={tuple(float(r[c]) for c in features) for r in candidates};assert len(distinct)==1,(key,frame,len(distinct))
   poses.append(np.array(next(iter(distinct))).reshape(8,2))
  groups[pred['actual']].append(poses);membership.append({**pred,'frames':[float(r['source_frame']) for r in seq]})
 arrays={k:np.array(v) for k,v in groups.items()};assert arrays['L'].shape==(15,3,8,2) and arrays['R'].shape==(9,3,8,2)
 assert all(np.isfinite(v).all() for v in arrays.values())
 mean={k:v.mean(axis=0) for k,v in arrays.items()};sd={k:v.std(axis=0,ddof=1) for k,v in arrays.items()}
 np.savez_compressed(OUT/'poses_mean_std.npz',L=arrays['L'],R=arrays['R'],mean_L=mean['L'],mean_R=mean['R'],std_L=sd['L'],std_R=sd['R'],joint_ids=JOINTS,frame_offsets=[-2,-1,0])
 with (OUT/'mean_std_coordinates.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['direction','n','frame_offset','joint_id','joint_name','x_mean','x_std','y_mean','y_std']);w.writeheader()
  for k in groups:
   for t in range(3):
    for j in range(8):w.writerow(dict(direction=k,n=len(arrays[k]),frame_offset=t-2,joint_id=JOINTS[j],joint_name=NAMES[j],x_mean=mean[k][t,j,0],x_std=sd[k][t,j,0],y_mean=mean[k][t,j,1],y_std=sd[k][t,j,1]))
 fig,axes=plt.subplots(1,3,figsize=(12,6),sharex=True,sharey=True)
 allmeans=np.concatenate(list(mean.values()));low=allmeans.min(axis=(0,1))-.22;high=allmeans.max(axis=(0,1))+.22
 for t,ax in enumerate(axes):
  for k in ['L','R']:
   p=mean[k][t]
   for a,b in EDGES:ax.plot(p[[a,b],0],p[[a,b],1],color=COLORS[k],lw=2.5)
   ax.scatter(p[:,0],p[:,1],color=COLORS[k],s=30,label=f'{k} correct (n={len(arrays[k])})')
   for j in range(8):ax.annotate(str(JOINTS[j]),p[j],xytext=(4,4 if k=='L' else -12),textcoords='offset points',fontsize=8,color=COLORS[k])
  ax.set_xlim(low[0],high[0]);ax.set_ylim(high[1],low[1]);ax.set_aspect('equal',adjustable='box');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.grid(alpha=.15);ax.set_title(['Contact - 2 frames','Contact - 1 frame','Contact frame'][t]);ax.set_xlabel('Normalized x (torso units)')
 axes[0].set_ylabel('Normalized y (image axis: downward)');axes[-1].legend(loc='lower right',fontsize=8)
 fig.suptitle('Mean 8-joint poses: correctly classified integrated test clips\nRight-foot canonical coordinates; shoulder joints are visualization only',fontsize=13)
 fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(OUT/'mean_skeletons.png',dpi=180);fig.savefig(OUT/'mean_skeletons.pdf');plt.close(fig)
 for axis,label in [(0,'x'),(1,'y')]:
  fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True)
  for j,ax in enumerate(axes.flat):
   for k in ['L','R']:
    m=mean[k][:,j,axis];s=sd[k][:,j,axis];ax.plot([-2,-1,0],m,'o-',color=COLORS[k],label=f'{k} (n={len(arrays[k])})');ax.fill_between([-2,-1,0],m-s,m+s,color=COLORS[k],alpha=.15)
   ax.set_title(f'{NAMES[j]} (kp {JOINTS[j]})');ax.set_xticks([-2,-1,0]);ax.grid(alpha=.2);ax.set_xlabel('Frame offset from contact');ax.set_ylabel(f'Normalized {label}')
  axes[0,0].legend();fig.suptitle(f'L/R mean {label}-coordinate curves, with +/-1 sample SD\nCorrectly classified test clips; equal weight per video',fontsize=13);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUT/f'joint_{label}_curves.png',dpi=180);fig.savefig(OUT/f'joint_{label}_curves.pdf');plt.close(fig)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 report={'passed':True,'input_hashes':hashes,'input_files_unchanged':True,'videos':24,'L':15,'R':9,'frames':[-2,-1,0],'joint_ids':JOINTS,'weighting':'Equal per correctly classified video','dispersion':'Sample standard deviation (ddof=1), not confidence interval','dataset_counts_by_label':{k:dict(Counter(r['dataset'] for r in membership if r['actual']==k)) for k in groups},'selected_videos':membership,'notes':['Shoulder coordinates recovered from original normalized full-sequence CSV; matched every selected lower-body coordinate. Model itself uses six joints only.','Descriptive correct-only curves; not feature attribution and not representative of all test videos.','No smoothing or temporal interpolation added; frame offsets are not equal physical durations across differing FPS.']}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));(OUT/'README.md').write_text('正答 L/R 8-joint curves\n\nCorrect integrated test only: L=15, R=9. Three original frames including contact. Equal per-video means and sample SD. Original pelvis/torso normalization, no extra reflection, alignment or rescaling. Shoulders are displayed but were not inputs to the six-joint classifier. All source files unchanged. See summary.json for selected clips and source distribution.\n\n[Mean skeletons](mean_skeletons.png) · [x curves](joint_x_curves.png) · [y curves](joint_y_curves.png) · [Mean and SD CSV](mean_std_coordinates.csv)\n')
 print(json.dumps({k:report[k] for k in ['passed','videos','L','R','dataset_counts_by_label']}))
if __name__=='__main__':main()
