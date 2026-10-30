"""Separate L/R mean eight-joint videos for correctly classified integrated-test clips."""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from visualize_penaltyfirst15_graphs import encode
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/experiment3_central_contact_correct_lr_20frame_videos'
JOINTS=[5,6,11,12,13,14,15,16]
EDGES=[(0,1),(0,2),(1,3),(2,3),(2,4),(4,6),(3,5),(5,7)]
def main():
 OUT.mkdir(exist_ok=False)
 source=ROOT/'Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled_with_150penalties.csv'
 pred=ROOT/'runs/experiment3_central_contact_pre3_seed42/test_integrated_predictions.csv'
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,pred]}
 correct=[r for r in csv.DictReader(pred.open()) if r['actual']==r['predicted']]
 keys={(r['dataset'],r['kick_id']):r['actual'] for r in correct}
 groups={k:[] for k in keys}
 for r in csv.DictReader(source.open()):
  k=(r['dataset'],r['kick_id'])
  if k in groups:groups[k].append(r)
 poses={'L':[],'R':[]}
 for key,rows in groups.items():
  rows.sort(key=lambda r:int(r['sample_20_idx']))
  assert len(rows)==20 and [int(r['sample_20_idx']) for r in rows]==list(range(20))
  assert rows[-1]['last_touch']=='1'
  poses[keys[key]].append(np.array([[[float(r[f'kp_{j}_xn']),float(r[f'kp_{j}_yn'])] for j in JOINTS] for r in rows]))
 arrays={k:np.array(v) for k,v in poses.items()};assert arrays['L'].shape==(15,20,8,2) and arrays['R'].shape==(9,20,8,2)
 mean={k:v.mean(axis=0) for k,v in arrays.items()};sd={k:v.std(axis=0,ddof=1) for k,v in arrays.items()}
 both=np.concatenate(list(mean.values()));low=both.min(axis=(0,1))-.25;high=both.max(axis=(0,1))+.25
 np.savez_compressed(OUT/'mean_20frame_poses.npz',L=arrays['L'],R=arrays['R'],mean_L=mean['L'],mean_R=mean['R'],std_L=sd['L'],std_R=sd['R'],joint_ids=JOINTS)
 with (OUT/'mean_20frame_coordinates.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['direction','n','sample_20_idx','joint_id','x_mean','y_mean','x_std','y_std']);w.writeheader()
  for k in mean:
   for t in range(20):
    for j in range(8):w.writerow(dict(direction=k,n=len(arrays[k]),sample_20_idx=t,joint_id=JOINTS[j],x_mean=mean[k][t,j,0],y_mean=mean[k][t,j,1],x_std=sd[k][t,j,0],y_std=sd[k][t,j,1]))
 for k,color in [('L','#2374b5'),('R','#e47b23')]:
  frames=[];gifs=[]
  fig,ax=plt.subplots(figsize=(7,8),dpi=120)
  for t in range(20):
   ax.clear();p=mean[k][t]
   for a,b in EDGES:ax.plot(p[[a,b],0],p[[a,b],1],color=color,lw=3)
   ax.scatter(p[:,0],p[:,1],c=color,s=50)
   for j in range(8):ax.annotate(str(JOINTS[j]),p[j],xytext=(5,5),textcoords='offset points',fontsize=10)
   ax.set_xlim(low[0],high[0]);ax.set_ylim(high[1],low[1]);ax.set_aspect('equal',adjustable='box');ax.axhline(0,c='gray',lw=.5);ax.axvline(0,c='gray',lw=.5);ax.grid(alpha=.2)
   ax.set_xlabel('Normalized x (torso units)');ax.set_ylabel('Normalized y (downward)')
   ax.set_title(f'{k} direction | mean of {len(arrays[k])} correct test clips\nSample {t+1}/20'+(' | CONTACT' if t==19 else ''),fontsize=15,color=color)
   fig.texts.clear();fig.text(.5,.025,'Start-to-contact progress alignment; right-foot canonical\nEqual video weights; fixed shared axes; playback slowed',ha='center',fontsize=10)
   fig.tight_layout(rect=[0,.065,1,1]);fig.canvas.draw();rgb=np.array(fig.canvas.buffer_rgba())[:,:,:3].copy();frames.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));gifs.append(Image.fromarray(rgb))
  encode(frames,OUT/f'{k}_mean_20frames.mp4')
  gifs[0].save(OUT/f'{k}_mean_20frames.gif',save_all=True,append_images=gifs[1:],duration=[250]*19+[1500],loop=0)
  gifs[-1].save(OUT/f'{k}_contact_mean.png');plt.close(fig)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 report={'passed':True,'L_videos':15,'R_videos':9,'mean_shape_per_direction':[20,8,2],'shared_bounds':[low.tolist(),high.tolist()],'selection':'Correct predictions from central-camera six-joint contact-inclusive 3-frame integrated test; same 24 clips as previous curves.','alignment':'20 uniformly sampled start-to-contact progress steps, not matching wall-clock times or consecutive source frames.','interpretation':'Eight-joint descriptive mean poses; six-joint model predictions unchanged. No extra interpolation or smoothing. Existing resampling, if any, retained from input CSV.','source_hashes':hashes,'selected_clips':correct,'dataset_counts':{k:dict(Counter(r['dataset'] for r in correct if r['actual']==k)) for k in poses}}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));(OUT/'README.md').write_text('L/R 별도 20프레임 평균 관절 영상\n\n정답 표본 L15/R9의 정규화된 8관절을 영상별 동일 가중치로 평균했습니다. 시작부터 접촉까지 균등 표집한 20개 진행 시점을 맞췄으며 실제 시간은 영상마다 다릅니다. L/R 영상의 축척은 동일합니다. 추가 보간·평활화 없이 기존 CSV를 사용했습니다. 표본 선정에 사용한 모델은 여전히 접촉 포함 3프레임·6관절 모델입니다. 영상 재생 속도는 시각화를 위한 느린 속도이며 실제 속도가 아닙니다.\n\n[L MP4](L_mean_20frames.mp4) · [R MP4](R_mean_20frames.mp4) · [L GIF](L_mean_20frames.gif) · [R GIF](R_mean_20frames.gif) · [좌표 CSV](mean_20frame_coordinates.csv)\n')
 print(json.dumps({'passed':True,'L':15,'R':9,'frames':20,'joints':8,'output':str(OUT)}))
if __name__=='__main__':main()
