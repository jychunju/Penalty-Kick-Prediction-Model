"""Render five random normalized skeletons on their shared pre-contact timeline."""
import csv
import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'penalty_kicks_20frames_combined_normalized.csv'
OUT = ROOT / 'runs/visualizations/random_5_normalized_overlap.mp4'
JOINTS = (11, 12, 13, 14, 15, 16)
EDGES = ((0,1),(0,2),(2,4),(1,3),(3,5))
COLORS = ((255,190,40),(50,210,255),(210,70,255),(90,240,100),(80,90,255))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-with-foot',action='store_true')
    args=parser.parse_args()
    input_path=(ROOT / 'penalty_kicks_20frames_combined_with_foot.csv') if args.raw_with_foot else INPUT
    output_path=(ROOT / 'runs/visualizations/random_5_raw_size_matched_overlap.mp4') if args.raw_with_foot else OUT
    with input_path.open(encoding='utf-8-sig', newline='') as f:
        groups = defaultdict(list)
        for row in csv.DictReader(f):
            groups[(row['dataset'],row['kick_id'])].append(row)
    eligible=[]
    for key, rows in groups.items():
        touches=[r for r in rows if r['last_touch'].strip() in ('1','1.0')]
        if len(touches)!=1: continue
        # Missing original joints are suppressed, not shown as real poses.
        clean=True
        for r in rows:
            for j in JOINTS:
                try:
                    xy=np.array([float(r[f'kp_{j}_x']),float(r[f'kp_{j}_y'])])
                    if not np.isfinite(xy).all() or (xy<=0).any():clean=False
                except ValueError:clean=False
        if args.raw_with_foot:
            try:
                fs=[float(r['frame']) for r in rows]
                clean=clean and len(set(fs))==len(fs)
            except ValueError:clean=False
        if clean:eligible.append(key)
    chosen=random.Random(42).sample(sorted(eligible),5)
    sequences=[]
    for key in chosen:
        frame_col='frame' if args.raw_with_foot else 'resolved_source_frame'
        rows=sorted(groups[key],key=lambda r:float(r[frame_col]))
        contact=next(float(r[frame_col]) for r in rows if r['last_touch'].strip() in ('1','1.0'))
        fps={'CLT':30.,'EPL':25.,'Women':22.12}[key[0]] if args.raw_with_foot else float(rows[0]['fps'])
        times=(np.array([float(r[frame_col]) for r in rows])-contact)/fps
        if args.raw_with_foot:
            poses=np.array([[[float(r[f'kp_{j}_{a}']) for a in 'xy'] for j in JOINTS] for r in rows])
            # Leg segment lengths respond less to knee bending than bbox height.
            lengths=(np.linalg.norm(poses[:,0]-poses[:,2],axis=1)+np.linalg.norm(poses[:,2]-poses[:,4],axis=1)
                     +np.linalg.norm(poses[:,1]-poses[:,3],axis=1)+np.linalg.norm(poses[:,3]-poses[:,5],axis=1))/2
            smooth=np.array([np.median(lengths[max(0,i-2):min(len(lengths),i+3)]) for i in range(len(lengths))])
            smooth=np.clip(smooth,np.median(lengths)*.65,np.median(lengths)*1.5)
            poses=(poses-((poses[:,0]+poses[:,1])/2)[:,None,:])/smooth[:,None,None]*.55
            if rows[0]['r_or_l']=='L':
                poses[...,0]*=-1
                poses=poses[:,[1,0,3,2,5,4]]
        else:
            poses=np.array([[[float(r[f'kp_{j}_{a}_norm']) for a in 'xy'] for j in JOINTS] for r in rows])
        sequences.append((key,times,poses))
    start=max(s[1][0] for s in sequences)
    timeline=np.linspace(start,0,max(2,int(round(-start*25))+1))
    interpolated=[]
    for key,times,poses in sequences:
        res=np.empty((len(timeline),6,2))
        for j in range(6):
            for a in range(2):res[:,j,a]=np.interp(timeline,times,poses[:,j,a])
        interpolated.append(res)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    temp=output_path.with_name(output_path.stem+'_mpeg4.mp4')
    size=900;scale=560;anchor=np.array([450,360])
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),12.5,(size,size))
    for t,relative in enumerate(timeline):
        im=np.full((size,size,3),(24,22,20),np.uint8)
        cv2.putText(im,'5 random kicks | right-foot normalized',(25,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(240,240,240),2,cv2.LINE_AA)
        for i,((key,_,_),poses,color) in enumerate(zip(sequences,interpolated,COLORS)):
            cv2.putText(im,f'{key[0]} / {key[1]}',(25,70+i*28),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
            pts=np.rint(poses[t]*scale+anchor).astype(int)
            for a,b in EDGES:cv2.line(im,tuple(pts[a]),tuple(pts[b]),color,3,cv2.LINE_AA)
            for p in pts:cv2.circle(im,tuple(p),5,color,-1,cv2.LINE_AA)
        cv2.putText(im,f't = {relative:.2f}s relative to contact | 0.5x',(25,850),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),2,cv2.LINE_AA)
        if t==len(timeline)-1:cv2.putText(im,'CONTACT',(350,805),cv2.FONT_HERSHEY_SIMPLEX,1,(80,80,255),2,cv2.LINE_AA)
        writer.write(im)
    writer.release()
    subprocess.run(['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temp),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(output_path)],check=True)
    meta={'input':str(input_path),'seed':42,'selected_kicks':[list(k) for k in chosen],
          'shared_relative_time_range_seconds':[float(start),0], 'frames':len(timeline),'output_fps':12.5,
          'size_matching':'framewise mean leg-chain length; 5-sample median smoothing; hip-centered; right-foot canonical' if args.raw_with_foot else 'existing normalized coordinates',
          'note':'Missing raw joints excluded; linear temporal interpolation for visualization; contact included.'}
    output_path.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
