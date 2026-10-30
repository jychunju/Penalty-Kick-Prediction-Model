"""Audit and report the three saved six-joint experiments without test reevaluation."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
NAMES=['uniform3','pre3','pre5']
TITLES=['Uniform 3 (start/middle/contact)','Precontact 3','Precontact 5']
OUT=ROOT/'runs/experiment3_central_contact_comparison'
def main():
 OUT.mkdir(exist_ok=True)
 metrics={};infos={};arrays={};splits={}
 original=json.loads((ROOT/'runs/experiment1_150penalties_stgcn_20f20features_seed42/split.json').read_text())
 for n in NAMES:
  folder=ROOT/f'runs/experiment3_central_contact_{n}_seed42';data=ROOT/f'data/experiment3_central_contact_{n}_seed42'
  m=json.loads((folder/'metrics.json').read_text());i=json.loads((folder/'run_info.json').read_text());sp=json.loads((folder/'split.json').read_text())
  a=np.load(data/'selected_windows.npz');x=a['X'];features=a['features'].tolist();expected=[f'kp_{j}_{axis}n' for j in [11,12,13,14,15,16] for axis in 'xy'];assert features==expected
  assert x.shape==(209,5 if n=='pre5' else 3,12) and np.isfinite(x).all()
  assert i['model']['parameters']==11066 and m['test_evaluations_per_set']=={'integrated':1,'CLT':1}
  assert [len(sp[k]) for k in ['train','validation','test','test_clt']]==[132,28,28,21]
  assert i['include_contact'] and not i['missing_precontact_exclusions']
  source_rows=list(csv.DictReader((data/'train.csv').open()))+list(csv.DictReader((data/'validation.csv').open()))+list(csv.DictReader((data/'test.csv').open()))
  assert all(r['camera_direction']=='C' and r['dataset']!='CLT' for r in source_rows)
  assert all(r['dataset']=='CLT' for r in csv.DictReader((data/'test_clt.csv').open()))
  assert hashlib.sha256(Path(i['input_csv']).read_bytes()).hexdigest()==i['input_sha256']
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in i['full_source_hashes'].items())
  for r in i['window_audit']:
   if n!='uniform3':assert r['frames']==list(range(r['contact_frame']-int(x.shape[1])+1,r['contact_frame']+1))
  for key in ['test_integrated','test_clt']:
   rows=list(csv.DictReader((folder/f'{key}_predictions.csv').open()));assert len(rows)==m[key]['num_kicks']
   assert sum(r['actual']==r['predicted'] for r in rows)/len(rows)==m[key]['accuracy']
  for key,indices in sp.items():
   rows=list(csv.DictReader((data/f'{key}_selected_windows.csv').open()));assert len(rows)==len(indices)*x.shape[1]
  metrics[n]=m;infos[n]=i;arrays[n]=x;splits[n]=sp
 assert splits['uniform3']==splits['pre3']==splits['pre5']
 assert np.array_equal(arrays['pre3'],arrays['pre5'][:,2:,:])
 audit={'passed':True,'same_splits_all_conditions':True,'central_camera_only':True, 'contact_included':True, 'new_source_stratified_split':True,'strict_consecutive_precontact_frames':True,'pre3_equals_final3_of_pre5':True,'features_12_xy_only':True,'all_finite':True,'input_files_unchanged':True,'test_evaluations_per_set_per_model':1,'parameters':11066}
 (OUT/'artifact_validation.json').write_text(json.dumps(audit,indent=2))
 (OUT/'comparison.json').write_text(json.dumps(metrics,indent=2))
 with (OUT/'comparison.csv').open('w') as f:
  fields=['condition','frames','best_epoch','train_accuracy','validation_accuracy','integrated_test_accuracy','integrated_test_macro_f1','CLT_test_accuracy','CLT_test_macro_f1'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for n,m in metrics.items():w.writerow(dict(condition=n,frames=5 if n=='pre5' else 3,best_epoch=m['best_epoch'],train_accuracy=m['train']['accuracy'],validation_accuracy=m['validation']['accuracy'],integrated_test_accuracy=m['test_integrated']['accuracy'],integrated_test_macro_f1=m['test_integrated']['macro_f1'],CLT_test_accuracy=m['test_clt']['accuracy'],CLT_test_macro_f1=m['test_clt']['macro_f1']))
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for j,n in enumerate(NAMES):
  for row,key in enumerate(['test_integrated','test_clt']):
   ax=axes[row,j];cm=np.array(metrics[n][key]['confusion_matrix_L_R']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=26);ax.set_xticks([0,1],['L','R']);ax.set_yticks([0,1],['L','R']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f"{n} / {'Integrated' if row==0 else 'CLT'} / {metrics[n][key]['accuracy']:.1%}")
   for a in range(2):
    for b in range(2):ax.text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]>13 else 'black')
 fig.tight_layout();fig.savefig(OUT/'confusion_matrices.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,n in zip(axes,NAMES):
  h=json.loads((ROOT/f'runs/experiment3_central_contact_{n}_seed42/history.json').read_text());ax.plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label='Train weighted loss');ax.plot([r['epoch'] for r in h],[r['validation']['loss'] for r in h],label='Validation loss');ax.axvline(metrics[n]['best_epoch'],ls='--',color='gray');ax.set_title(n);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
 lines=['# Central-camera six-joint ST-GCN: contact-inclusive 3/5-frame windows','', 'L/R binary classification; seed 42; camera C only (CLT=C by prior user confirmation). CLT external only; no missing-window exclusions: train 132, validation 28, integrated test 28, CLT test 21. New camera-C source-stratified 70/15/15 split; identical in every condition. No missing frame is synthesized.','', 'Original 75.02% ± 8.77 percentage points was 5-fold × 3-seed validation on C-camera development data, including CLT. It is not a benchmark on these new central-camera test sets. Architecture and training hyperparameters reused; population and evaluation protocol differ.','', 'Model: hips/knees/ankles (COCO 11–16), 12 XY coordinates only, no angles. ST-GCN 2→16→16→32; 11066 parameters; AdamW lr=.001, decay=.001, batch16, dropout=.3, maximum150 epochs, patience25, training-only inverse class weights. Model selection uses validation macro F1 then loss; two test sets evaluated once per model. Original pelvis/full-sequence torso normalization retained.','', 'uniform3 selects 20-frame sample indices [0,10,19] including contact. pre3/pre5 select original frames contact−2..contact / contact−4..contact INCLUDING contact.','', '| Condition | Train acc | Valid acc | Integrated test acc | Integrated F1 | CLT test acc | CLT F1 | Best epoch |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for n,m in metrics.items():lines.append(f"| {n} | {m['train']['accuracy']:.2%} | {m['validation']['accuracy']:.2%} | {m['test_integrated']['accuracy']:.2%} | {m['test_integrated']['macro_f1']:.4f} | {m['test_clt']['accuracy']:.2%} | {m['test_clt']['macro_f1']:.4f} | {m['best_epoch']} |")
 lines+=['','## Integrated test by source','','| Condition | EPL (n=4) | Women (n=18) | 150penalties (n=6) |','|---|---:|---:|---:|']
 for n,m in metrics.items():lines.append('| '+n+' | '+' | '.join(f"{m['test_integrated']['by_dataset'][d]['accuracy']:.2%}" for d in ['EPL','Women','150penalties'])+' |')
 lines+=['','[Confusion matrices](confusion_matrices.png) · [Learning curves](learning_curves.png) · [Full metrics](comparison.json) · [Artifact validation](artifact_validation.json)','', 'Per-condition run folders contain best.pt, history.json, metrics.json, split.json and two prediction CSVs. In data folders, selected_windows.npz and *_selected_windows.csv are actual model inputs; samples.npz and train/validation/test CSVs retain the original 20-frame source inputs for provenance. Window/frame audit is in run_info.json.','', 'Limitations: single seed, small tests, no full player/match separation, assumed contact for 67 new clips. Precontact temporal duration varies with FPS. Every window includes contact and full-sequence normalization: these tests do not demonstrate advance prediction before contact. Precontact 033 uses original consecutive observations, whereas its uniform3 input derives from the earlier 18→20 temporal resampling. Do not choose/tune future models using these test results.','']
 (OUT/'README.md').write_text('\n'.join(lines));print(json.dumps({n:{'best_epoch':m['best_epoch'],'epochs':m['epochs_completed'],'train':m['train']['accuracy'],'validation':m['validation']['accuracy'],'integrated':m['test_integrated']['accuracy'],'integrated_f1':m['test_integrated']['macro_f1'],'clt':m['test_clt']['accuracy'],'clt_f1':m['test_clt']['macro_f1']} for n,m in metrics.items()},indent=2))
if __name__=='__main__':main()
