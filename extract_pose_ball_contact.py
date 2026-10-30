#!/usr/bin/env python3
"""Extract a kicker pose, detect the ball, estimate contact, and render 0.5x video."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def args():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pose-model", default="yolov8m-pose.pt")
    p.add_argument("--detect-model", default="yolov8m.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--pose-conf", type=float, default=.20)
    p.add_argument("--ball-conf", type=float, default=.08)
    return p.parse_args()


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened(): raise FileNotFoundError(a.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames=[]
    while True:
        ok, frame=cap.read()
        if not ok: break
        frames.append(frame)
    cap.release()
    if not frames: raise RuntimeError("No frames")

    pose_model=YOLO(a.pose_model); ball_model=YOLO(a.detect_model)
    people=[]; balls=[]
    for fi, frame in enumerate(frames):
        pr=pose_model.track(frame,persist=True,tracker="bytetrack.yaml",device=a.device,imgsz=a.imgsz,conf=a.pose_conf,verbose=False)[0]
        frame_people=[]
        if pr.boxes is not None and pr.keypoints is not None and len(pr.boxes):
            boxes=pr.boxes.xyxy.cpu().numpy(); bconf=pr.boxes.conf.cpu().numpy()
            ids=pr.boxes.id.cpu().numpy().astype(int) if pr.boxes.id is not None else np.full(len(boxes),-1)
            xy=pr.keypoints.xy.cpu().numpy(); kc=pr.keypoints.conf.cpu().numpy()
            for box,conf,tid,kpts,kconf in zip(boxes,bconf,ids,xy,kc):
                frame_people.append({"track_id":int(tid),"box":box,"conf":float(conf),"xy":kpts,"kconf":kconf})
        people.append(frame_people)
        br=ball_model.predict(frame,device=a.device,imgsz=a.imgsz,conf=a.ball_conf,classes=[32],verbose=False)[0]
        candidates=[]
        if br.boxes is not None:
            for box,conf in zip(br.boxes.xyxy.cpu().numpy(),br.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2=map(float,box); candidates.append({"xy":np.array([(x1+x2)/2,(y1+y2)/2]),"box":box,"conf":float(conf)})
        # The highest confidence sports-ball detection is retained; raw candidates remain auditable in CSV.
        balls.append(max(candidates,key=lambda x:x["conf"]) if candidates else None)

    # Select the track that is persistent and approaches the detected ball with an ankle.
    tids=sorted({p["track_id"] for pp in people for p in pp if p["track_id"]>=0})
    scores=[]
    for tid in tids:
        obs=[]; near=[]
        for fi,pp in enumerate(people):
            p=next((x for x in pp if x["track_id"]==tid),None)
            if p is None: continue
            obs.append(fi)
            if balls[fi] is not None:
                valid=[j for j in (15,16) if p["kconf"][j]>=.15 and np.all(p["xy"][j]>0)]
                if valid:
                    d=min(np.linalg.norm(p["xy"][j]-balls[fi]["xy"]) for j in valid)
                    near.append(d/max(float(p["box"][3]-p["box"][1]),1.0))
        score=len(obs)/len(frames) - (min(near) if near else 10.0)
        scores.append((score,tid,len(obs),min(near) if near else None))
    if not scores: raise RuntimeError("No tracked people")
    scores.sort(reverse=True); kicker_id=scores[0][1]

    # Fill only short internal gaps in ball centers for distance evaluation.
    ball_xy=np.full((len(frames),2),np.nan)
    ball_interpolated=np.zeros(len(frames),bool)
    for i,b in enumerate(balls):
        if b is not None: ball_xy[i]=b["xy"]
    valid=np.flatnonzero(np.isfinite(ball_xy[:,0]))
    for left,right in zip(valid[:-1],valid[1:]):
        if 1 < right-left <= 4:
            for i in range(left+1,right):
                q=(i-left)/(right-left);ball_xy[i]=(1-q)*ball_xy[left]+q*ball_xy[right];ball_interpolated[i]=True

    distances=np.full(len(frames),np.nan); ankle_used=np.full(len(frames),-1,int)
    chosen=[]
    for fi,pp in enumerate(people):
        p=next((x for x in pp if x["track_id"]==kicker_id),None);chosen.append(p)
        if p is None or not np.isfinite(ball_xy[fi]).all(): continue
        opts=[]
        for j in (15,16):
            if p["kconf"][j]>=.15 and np.all(p["xy"][j]>0): opts.append((np.linalg.norm(p["xy"][j]-ball_xy[fi]),j))
        if opts:
            distances[fi],ankle_used[fi]=min(opts)
    usable=np.flatnonzero(np.isfinite(distances))
    if len(usable):
        # Contact must occur after the midpoint of these short, pre-contact clips.
        latter=usable[usable>=len(frames)//2]; pool=latter if len(latter) else usable
        contact=int(pool[np.argmin(distances[pool])])
    else: contact=None

    csv_path=a.output_dir/"pose_ball_contact.csv"
    fields=["frame","time_sec","track_id","person_conf","ball_x","ball_y","ball_conf","ball_interpolated","nearest_ankle","ankle_ball_distance_px","contact_candidate"]
    for name in NAMES: fields += [f"{name}_x",f"{name}_y",f"{name}_conf"]
    with csv_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for fi,p in enumerate(chosen):
            b=balls[fi]; row={"frame":fi,"time_sec":fi/fps,"track_id":kicker_id,"person_conf":"" if p is None else p["conf"],"ball_x":"" if not np.isfinite(ball_xy[fi,0]) else ball_xy[fi,0],"ball_y":"" if not np.isfinite(ball_xy[fi,1]) else ball_xy[fi,1],"ball_conf":"" if b is None else b["conf"],"ball_interpolated":int(ball_interpolated[fi]),"nearest_ankle":{15:"left",16:"right"}.get(ankle_used[fi],""),"ankle_ball_distance_px":"" if not np.isfinite(distances[fi]) else distances[fi],"contact_candidate":int(fi==contact)}
            for j,name in enumerate(NAMES):
                row[f"{name}_x"]="" if p is None else p["xy"][j,0];row[f"{name}_y"]="" if p is None else p["xy"][j,1];row[f"{name}_conf"]="" if p is None else p["kconf"][j]
            w.writerow(row)

    temp=a.output_dir/"pose_ball_contact_0.5x_mpeg4.mp4";output=a.output_dir/"pose_ball_contact_0.5x.mp4"
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*"mp4v"),fps*.5,(width,height))
    for fi,(im,p) in enumerate(zip(frames,chosen)):
        im=im.copy()
        if p is not None:
            pts=[None if p["kconf"][j]<.15 or not np.all(p["xy"][j]>0) else tuple(np.rint(p["xy"][j]).astype(int)) for j in range(17)]
            for j,k in EDGES:
                if pts[j] is not None and pts[k] is not None:cv2.line(im,pts[j],pts[k],(0,255,255),4,cv2.LINE_AA)
            for point in pts:
                if point is not None:cv2.circle(im,point,5,(0,0,255),-1,cv2.LINE_AA)
            x1,y1,x2,y2=np.rint(p["box"]).astype(int);cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
        if np.isfinite(ball_xy[fi]).all():
            center=tuple(np.rint(ball_xy[fi]).astype(int)); color=(0,165,255) if ball_interpolated[fi] else (255,0,255)
            cv2.circle(im,center,14,color,3,cv2.LINE_AA);cv2.putText(im,"BALL"+(" interp" if ball_interpolated[fi] else ""),(center[0]+16,center[1]-10),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
        cv2.putText(im,f"frame {fi}/{len(frames)-1}  t={fi/fps:.2f}s  playback=0.5x",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
        if np.isfinite(distances[fi]):cv2.putText(im,f"ankle-ball: {distances[fi]:.1f}px",(20,68),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        if fi==contact:
            cv2.rectangle(im,(0,height-75),(width,height),(0,0,180),-1);cv2.putText(im,"CONTACT CANDIDATE (minimum ankle-ball distance)",(20,height-25),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2,cv2.LINE_AA)
        writer.write(im)
    writer.release()
    subprocess.run(["/usr/bin/ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(temp),"-c:v","libx264","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],check=True)
    meta={"input":str(a.video.resolve()),"source_fps":fps,"output_fps":fps*.5,"playback_speed":.5,"frames":len(frames),"pose_model":a.pose_model,"ball_model":a.detect_model,"kicker_track_id":kicker_id,"track_ranking":[{"score":x[0],"track_id":x[1],"frames":x[2],"min_normalized_ankle_ball_distance":x[3]} for x in scores],"ball_direct_detection_frames":int(sum(b is not None for b in balls)),"ball_interpolated_frames":int(ball_interpolated.sum()),"contact_candidate_frame":contact,"contact_candidate_sec":None if contact is None else contact/fps,"contact_distance_px":None if contact is None else float(distances[contact]),"contact_method":"minimum detected/interpolated ball-to-visible-ankle distance in latter half","contact_validation":"automatic candidate; visual review required, especially when post-contact frames are absent","outputs":{"csv":str(csv_path),"video":str(output)}}
    (a.output_dir/"metadata.json").write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
