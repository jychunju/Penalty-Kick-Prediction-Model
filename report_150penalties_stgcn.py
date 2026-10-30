"""Summarize experiment 1 from saved predictions/history, without rerunning tests."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/experiment1_150penalties_stgcn_20f20features_seed42'
def main():
 m=json.loads((OUT/'metrics.json').read_text());info=json.loads((OUT/'run_info.json').read_text());h=json.loads((OUT/'history.json').read_text())
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 e=[r['epoch'] for r in h];axes[0].plot(e,[r['train_loss'] for r in h],label='Train weighted loss');axes[0].plot(e,[r['validation']['loss'] for r in h],label='Validation loss')
 axes[1].plot(e,[r['validation']['accuracy'] for r in h],label='Validation accuracy');axes[1].plot(e,[r['validation']['macro_f1'] for r in h],label='Validation macro F1')
 for ax in axes:ax.axvline(m['best_epoch'],ls='--',color='gray',label='Selected epoch');ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(9,4))
 for ax,key,title in zip(axes,['test_integrated','test_clt'],['Integrated test','CLT external test']):
  a=np.array(m[key]['confusion_matrix_L_R']);ax.imshow(a,cmap='Blues',vmin=0,vmax=max(a.max(),1));ax.set_xticks([0,1],['L','R']);ax.set_yticks([0,1],['L','R']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f"{title} (n={a.sum()})")
  for i in range(2):
   for j in range(2):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if a[i,j]>a.max()/2 else 'black',fontsize=17)
 fig.tight_layout();fig.savefig(OUT/'test_confusion_matrices.png',dpi=170);plt.close(fig)
 lines=['# Experiment 1: 20 frames / 20 features ST-GCN','', 'L/R binary classification, seed 42. C directions (28 clips) excluded. CLT never enters training, validation or the integrated test. All frames of each video remain together. Dataset proportions are conserved by optimal integer quotas; exact equal proportions are impossible at these sample sizes.','', '| Split | EPL | Women | 150penalties | CLT | Total |','|---|---:|---:|---:|---:|---:|']
 for k in ['train','validation','test','test_clt']:
  ds=info[k]['datasets'];lines.append(f"| {k} | {ds.get('EPL',0)} | {ds.get('Women',0)} | {ds.get('150penalties',0)} | {ds.get('CLT',0)} | {info[k]['kicks']} |")
 lines+=['','Input: (batch,20,20). Features: 16 normalized coordinates for eight joints plus four normalized angles. ST-GCN coordinate branch (2→32→32→64 channels) and a temporal angle branch (4→16→16), followed by an 80→2 classifier. Metadata never enters the model.','', 'AdamW, lr=0.001, weight_decay=0.001, batch=16, dropout=0.3, train-only inverse-frequency class weights. Maximum 150 epochs, patience=30. Validation macro F1, then loss, selects the checkpoint. Each test evaluated once after selection.','',f"Selected epoch: {m['best_epoch']}; completed epochs: {m['epochs_completed']}.",'','| Evaluation | N | Accuracy | Macro F1 | Recall L | Recall R |','|---|---:|---:|---:|---:|---:|']
 for k in ['train','validation','test_integrated','test_clt']:
  r=m[k];lines.append(f"| {k} | {r['num_kicks']} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} | {r['recall_L_R'][0]:.2%} | {r['recall_L_R'][1]:.2%} |")
 lines+=['','## Integrated test by dataset','','| Dataset | N | Accuracy | Macro F1 |','|---|---:|---:|---:|']
 for ds,r in m['test_integrated']['by_dataset'].items():lines.append(f"| {ds} | {r['num_kicks']} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} |")
 lines+=['','[Learning curves](learning_curves.png) · [Confusion matrices](test_confusion_matrices.png) · [Metrics](metrics.json) · [Checkpoint](best.pt)','', 'Limitations: a single exploratory seed, small test sets, no player/match-level separation because complete identities are unavailable. The new data include assumed end-of-clip contact markers and one temporally resampled clip. Full contact-inclusive input is a baseline, not an early prediction experiment. CLT differences cannot be attributed to amateur skill alone; camera and extraction/domain differences also matter.','']
 (OUT/'README.md').write_text('\n'.join(lines));print(json.dumps({k:m[k] for k in ['best_epoch','epochs_completed','test_integrated','test_clt']},indent=2))
if __name__=='__main__':main()
