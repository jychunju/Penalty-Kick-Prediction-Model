#!/usr/bin/env python3
"""Run the existing first15 YOLOv8m pose pipeline over the reviewed 150penalties manifest.

Four local workers use separate GPUs for inference. Segment endpoints and kicker
seeds are explicitly recorded in segments.json; missing joints remain missing.
"""
import argparse,csv,json,os,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import extract_penaltyfirst15_pose as core
from reviewed_pose_inference import infer as reviewed_infer

BASE=Path(__file__).resolve().parent
INPUT=BASE/'Main Data/Video Data/150penalties'
OUTPUT=BASE/'runs/150penalties_pose'

def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=Path,default=BASE/'annotations/150penalties_segments.json')
 p.add_argument('--stage',choices=['infer','render','combine','all'],default='all')
 p.add_argument('--input-dir',type=Path,default=INPUT)
 p.add_argument('--output-dir',type=Path,default=OUTPUT)
 p.add_argument('--workers',type=int,default=4)
 p.add_argument('--worker',type=int)
 p.add_argument('--force',action='store_true')
 p.add_argument('--only-video')
 p.add_argument('--pose-model',default=str(BASE/'yolov8m-pose.pt'))
 p.add_argument('--ball-model',default=str(BASE/'yolov8m.pt'))
 p.add_argument('--imgsz',type=int,default=960)
 p.add_argument('--batch',type=int,default=8)
 p.add_argument('--keypoint-conf',type=float,default=.25)
 return p.parse_args()

def update_metadata(dest,spec):
 meta_path=dest/"metadata.json"
 meta=json.loads(meta_path.read_text())
 if spec.get("association_anchors"):
  meta["association"]="Visually reviewed position anchors plus temporal center continuity select YOLO detections within ROI; joint coordinates are not interpolated."
 if spec.get("rejected_detection_frames"):
  meta["manual_detection_rejection"]=spec["rejected_detection_frames"]
 meta_path.write_text(json.dumps(meta,indent=2))
 return meta

def worker(a,manifest):
 core.torch.set_num_threads(4);core.cv2.setNumThreads(2)
 original_run=subprocess.run
 def limited_run(cmd,*args,**kwargs):
  if isinstance(cmd,list) and Path(cmd[0]).name=='ffmpeg':
   cmd=cmd[:1]+['-threads','2','-filter_threads','1','-filter_complex_threads','1']+cmd[1:-1]+['-threads','2']+cmd[-1:]
  return original_run(cmd,*args,**kwargs)
 core.subprocess.run=limited_run
 a.device=str(a.worker)
 models=None
 names=[n for j,n in enumerate(manifest) if j%a.workers==a.worker and (not a.only_video or n==a.only_video)]
 for name in names:
  spec=manifest[name];dest=a.output_dir/Path(name).stem;dest.mkdir(parents=True,exist_ok=True)
  if a.stage=='infer':
   if (dest/'inference_all_frames.npz').exists() and not a.force:
    print(f'CACHE {name}',flush=True);continue
   if models is None:models=(core.YOLO(a.pose_model),core.YOLO(a.ball_model))
   print(f'INFER {name}',flush=True);reviewed_infer(a.input_dir/name,spec,dest,*models,a)
  elif a.stage=='render':
   meta=dest/'metadata.json'
   if meta.exists() and not a.force:
    previous=json.loads(meta.read_text())
    if all(previous.get(k)==v for k,v in spec.items() if k!='contact_frame') and previous.get('segment_end_frame')==spec['contact_frame']:
     print(f'EXISTS {name}',flush=True);continue
   print(f'RENDER {name}',flush=True);core.render(a.input_dir/name,spec,dest,a);update_metadata(dest,spec)
 print(f'WORKER {a.worker} DONE',flush=True)

def launch(a,stage):
 processes=[]
 for i in range(a.workers):
  log=(a.output_dir/f'{stage}_worker_{i}.log').open('w')
  cmd=[sys.executable,str(Path(__file__).resolve()),'--stage',stage,'--worker',str(i),'--workers',str(a.workers),'--input-dir',str(a.input_dir),'--output-dir',str(a.output_dir),'--pose-model',a.pose_model,'--ball-model',a.ball_model,'--imgsz',str(a.imgsz),'--batch',str(a.batch),'--keypoint-conf',str(a.keypoint_conf)]
  if a.force:cmd.append('--force')
  if a.only_video:cmd+=['--only-video',a.only_video]
  processes.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log,i))
 failures=[]
 for process,log,i in processes:
  code=process.wait();log.close()
  if code:failures.append((i,code))
 if failures:raise RuntimeError(f'{stage} workers failed: {failures}; see worker logs')

def combine(a,manifest):
 summaries=[];rows=[];graphs=[]
 for name in manifest:
  dest=a.output_dir/Path(name).stem
  summaries.append(update_metadata(dest,manifest[name]))
  with (dest/'kicker_keypoints_17j.csv').open(newline='') as f:rows.extend(csv.DictReader(f))
  with (dest/'kicker_graph_17j.csv').open(newline='') as f:graphs.extend(csv.DictReader(f))
 core.write_csv(a.output_dir/'all_kicker_keypoints_17j.csv',rows)
 core.write_csv(a.output_dir/'all_kicker_graph_17j.csv',graphs)
 (a.output_dir/'summary.json').write_text(json.dumps(summaries,indent=2))
 core.write_csv(a.output_dir/'quality_summary.csv',[{k:m[k] for k in ('video','runup_start_frame','contact_frame','segment_end_frame','extracted_frames','contact_status','pose_detected_frames','joint_valid_fraction','lower_body_valid_fraction','contact_valid_joints')} for m in summaries])
 (a.output_dir/'graph_schema.json').write_text(json.dumps({'joint_names':list(core.NAMES),'edge_index':core.EDGES.T.tolist(),'layout':'frames x 17 joints x 2 coordinates','confidence_layout':'frames x 17','source_frame_indexing':'zero-based, end inclusive','source_documentation':'https://docs.ultralytics.com/tasks/pose/'},indent=2))
 clips=[a.output_dir/Path(n).stem/'kicker_review_0.5x.mp4' for n in manifest]
 concat=a.output_dir/'review_concat.txt'
 concat.write_text(''.join("file '"+str(p.resolve()).replace("'","'\\''")+"'\n" for p in clips))
 subprocess.run([core.FFMPEG,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(a.output_dir/'all_kickers_review_0.5x.mp4')],check=True)
 for source,target in [('all_kicker_keypoints_17j.csv','150penalties_pose_keypoints.csv'),('all_kicker_graph_17j.csv','150penalties_pose_graphs.csv'),('summary.json','150penalties_pose_summary.json')]:
  shutil.copyfile(a.output_dir/source,BASE/'Main Data'/target)
 print(f'COMBINED {len(manifest)} videos, {len(rows)} frames',flush=True)

def main():
 a=arguments();a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(a.manifest.read_text())
 assert set(manifest)=={p.name for p in a.input_dir.glob('*.mp4')}
 if a.worker is not None:worker(a,manifest);return
 if a.stage in ('infer','all'):launch(a,'infer')
 if a.stage in ('render','all'):launch(a,'render')
 if a.stage in ('combine','all'):combine(a,manifest)

if __name__=='__main__':main()
