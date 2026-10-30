"""Integrated gradients and mean-replacement sensitivity for saved central pre3 model."""
from pathlib import Path
import csv,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_frames_joints import FlexiblePoseClassifier
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/experiment3_central_contact_joint_importance'
JOINTS=[11,12,13,14,15,16]
NAMES=['Left hip','Right hip','Left knee','Right knee','Left ankle','Right ankle']
def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
 run=ROOT/'runs/experiment3_central_contact_pre3_seed42';data=ROOT/'data/experiment3_central_contact_pre3_seed42'
 d=np.load(data/'selected_windows.npz');base=np.load(data/'samples.npz');X=torch.tensor(d['X']);y=base['y'];split=base['split'];ids=d['group_id'].tolist()
 model=FlexiblePoseClassifier('stgcn',6,.3);ck=torch.load(run/'best.pt',map_location='cpu');model.load_state_dict(ck['model']);model.eval()
 train=split=='train';test=np.flatnonzero(split=='test');x=X[test];target=torch.tensor(y[test]);baseline=X[train].mean(0,keepdim=True)
 with torch.no_grad(): logits=model(x);pred=logits.argmax(1);correct=pred==target
 assert int(correct.sum())==24
 # Signed attribution of the true-class logit margin versus the other class.
 def margin(z):
  out=model(z);return out.gather(1,target[:,None]).squeeze(1)-out.gather(1,(1-target)[:,None]).squeeze(1)
 def ig(steps):
  total=torch.zeros_like(x)
  for i in range(steps+1):
   z=(baseline+(i/steps)*(x-baseline)).detach().requires_grad_(True)
   g=torch.autograd.grad(margin(z).sum(),z)[0];total+=g*(.5 if i in (0,steps) else 1.)
  return ((x-baseline)*total/steps).detach()
 with torch.no_grad():difference=margin(x)-margin(baseline.expand_as(x))
 steps=256
 while True:
  attribution=ig(steps);error=attribution.sum((1,2))-difference
  relative=abs(error)/(1+abs(difference))
  if float(relative.max())<.01 or steps>=2048:break
  steps*=2
 assert torch.isfinite(attribution).all()
 signed=attribution.numpy().reshape(28,3,6,2).sum(-1)
 magnitude=abs(attribution.numpy()).reshape(28,3,6,2).sum(-1)
 masks={'Correct all':correct.numpy(),'Correct L':correct.numpy()&(target.numpy()==0),'Correct R':correct.numpy()&(target.numpy()==1),'All test':np.ones(28,dtype=bool)}
 with torch.no_grad():
  sensitivity=[]
  for j in range(6):
   changed=x.clone();changed[:,:,2*j:2*j+2]=baseline[:,:,2*j:2*j+2]
   sensitivity.append((margin(x)-margin(changed)).numpy())
 sensitivity=np.array(sensitivity).T
 summaries={}
 for name,mask in masks.items():
  mag=magnitude[mask].mean(0);sig=signed[mask].mean(0);joint=mag.sum(0);share=100*joint/joint.sum()
  summaries[name]={'n':int(mask.sum()),'joint_importance':joint.tolist(),'joint_share_percent':share.tolist(),'mean_signed_time_joint':sig.tolist(),'mean_absolute_time_joint':mag.tolist(),'mean_replacement_margin_drop':sensitivity[mask].mean(0).tolist()}
 with (OUT/'joint_ranking.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['subset','n','rank','joint_id','joint_name','mean_abs_IG','share_percent','mean_replacement_margin_drop']);w.writeheader()
  for name,r in summaries.items():
   for rank,j in enumerate(np.argsort(-np.array(r['joint_importance'])),1):w.writerow(dict(subset=name,n=r['n'],rank=rank,joint_id=JOINTS[j],joint_name=NAMES[j],mean_abs_IG=r['joint_importance'][j],share_percent=r['joint_share_percent'][j],mean_replacement_margin_drop=r['mean_replacement_margin_drop'][j]))
 with (OUT/'per_video_attribution.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['group_id','actual','correct','frame_offset','joint_id','signed_IG','absolute_IG']);w.writeheader()
  for i,idx in enumerate(test):
   for t in range(3):
    for j in range(6):w.writerow(dict(group_id=ids[idx],actual=['L','R'][y[idx]],correct=bool(correct[i]),frame_offset=t-2,joint_id=JOINTS[j],signed_IG=float(signed[i,t,j]),absolute_IG=float(magnitude[i,t,j])))
 fig,axes=plt.subplots(1,3,figsize=(14,5))
 keys=['Correct all','Correct L','Correct R'];vmax=max(np.max(summaries[k]['mean_absolute_time_joint']) for k in keys)
 for ax,k in zip(axes,keys):
  a=np.array(summaries[k]['mean_absolute_time_joint']);im=ax.imshow(a,cmap='YlOrRd',vmin=0,vmax=vmax,aspect='auto');ax.set_xticks(range(6),NAMES,rotation=50,ha='right');ax.set_yticks(range(3),['-2','-1','0 (contact)']);ax.set_title(f"{k} (n={summaries[k]['n']})");ax.set_ylabel('Frame offset' if k=='Correct all' else '')
  for t in range(3):
   for j in range(6):ax.text(j,t,f'{a[t,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if a[t,j]>vmax*.6 else 'black')
 fig.suptitle('Mean absolute Integrated Gradients | true-class logit margin\nTrain-mean baseline; shared color scale; 6 model-input joints',fontsize=12);fig.tight_layout(rect=[0,0,1,.88]);fig.colorbar(im,ax=axes.tolist(),shrink=.65,pad=.02);fig.savefig(OUT/'time_joint_heatmaps.png',dpi=180,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,5),sharex=True)
 for ax,k in zip(axes,keys):
  r=summaries[k];order=np.argsort(r['joint_share_percent']);ax.barh(np.array(NAMES)[order],np.array(r['joint_share_percent'])[order],color='#2374b5' if k!='Correct R' else '#e47b23');ax.set_title(f"{k} (n={r['n']})");ax.set_xlabel('Share of absolute IG (%)');ax.grid(axis='x',alpha=.2)
 fig.suptitle('Joint importance ranking (x/y magnitudes summed across 3 frames)');fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUT/'joint_importance.png',dpi=180);plt.close(fig)
 np.savez_compressed(OUT/'attributions.npz',attribution=attribution.numpy(),signed=signed,magnitude=magnitude,test_group_ids=np.array([ids[i] for i in test]),correct=correct.numpy(),target=target.numpy(),baseline=baseline.numpy(),completeness_error=error.numpy())
 report={'model_epoch':ck['epoch'],'method':'Trapezoidal integrated gradients of true-class logit margin (true minus other)','baseline':'Mean of training inputs at each time/coordinate; 132 train clips only','steps':steps,'max_completeness_error':float(abs(error).max()),'max_scaled_completeness_error':float(relative.max()),'summaries':summaries,'limitations':['Importance is baseline-dependent model attribution, not causal biomechanics.','Shoulders and angles are absent from this model and have no model-input attribution.','Mean replacement can produce uncommon poses; use as sensitivity only.','Correct-only samples are selected; all-test results provided separately.']}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));(OUT/'README.md').write_text('6관절 모델 중요도\n\n정답 L15/R9를 주 시각화 대상으로 하고 전체 test28 결과도 CSV에 제공합니다. train132의 시점별 평균 입력을 기준으로 실제 클래스 대 다른 클래스 logit 차이에 대한 Integrated Gradients를 계산했습니다. 중요도는 좌표 기여도의 절댓값을 x/y 및 3시점에 합산하며, 히트맵은 같은 색 범위를 사용합니다. signed 기여도는 실제 클래스 방향을 지지하면 양수입니다. train 평균으로 관절을 대체하는 민감도도 함께 제공합니다. 어깨와 각도는 모델 입력이 아니므로 중요도를 산출하지 않았습니다. 이는 모델 의존도 분석이며 인과적 동작 중요도가 아닙니다.\n')
 print(json.dumps({'steps':steps,'max_scaled_error':float(relative.max()),'correct_all':summaries['Correct all']},indent=2))
if __name__=='__main__':main()
