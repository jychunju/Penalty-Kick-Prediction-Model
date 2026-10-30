#!/usr/bin/env python3
"""Overlay two hip-centered COCO pose sequences, aligned at last_touch."""

from __future__ import annotations

import argparse,csv,json,subprocess
from pathlib import Path
import cv2
import numpy as np

JOINTS=tuple(range(5,17))
EDGES=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
MIRROR_JOINT={5:6,6:5,7:8,8:7,9:10,10:9,11:12,12:11,13:14,14:13,15:16,16:15}

def load(path,kick,camera):
    with path.open(newline="",encoding="utf-8-sig") as f:
        rows=[{k:(v or "").strip() for k,v in r.items()} for r in csv.DictReader(f)
              if (r.get("kick_id") or "").strip()==kick and (r.get("camera_direction") or "").strip()==camera]
    groups={(r.get("camera_direction"),r.get("track_id")) for r in rows}
    if len(groups)!=1:raise ValueError(f"kick {kick}, camera {camera}: expected one track, got {groups}")
    rows.sort(key=lambda r:int(float(r["frame"])))
    touches=[i for i,r in enumerate(rows) if r.get("last_touch")=="1"]
    if len(touches)!=1:raise ValueError(f"kick {kick}: expected one last_touch")
    feet={r.get("r_or_l","").upper() for r in rows if r.get("r_or_l","")}
    if len(feet)!=1 or next(iter(feet)) not in {"L","R"}:raise ValueError(f"kick {kick}: invalid kicking foot {feet}")
    return rows,touches[0],next(iter(groups))[1],next(iter(feet))

def points(row,size,scale,mirror=False):
    raw={}
    for j in JOINTS:
        try:
            xy=np.asarray([float(row[f"kp_{j}_x"]),float(row[f"kp_{j}_y"])])
            if np.isfinite(xy).all():raw[j]=xy
        except ValueError:pass
    if 11 in raw and 12 in raw:center=(raw[11]+raw[12])/2
    else:center=np.asarray([float(row["bbox_cx"]),float(row["bbox_cy"])])
    anchor=np.asarray([size*.5,size*.42])
    centered={j:(xy-center).copy() for j,xy in raw.items()}
    if mirror:
        for xy in centered.values():xy[0]*=-1
        centered={MIRROR_JOINT[j]:xy for j,xy in centered.items()}
    return {j:np.rint(xy*scale+anchor).astype(int) for j,xy in centered.items()}

def draw(image,pts,color):
    for a,b in EDGES:
        if a in pts and b in pts:cv2.line(image,tuple(pts[a]),tuple(pts[b]),color,5,cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(image,tuple(p),8,color,-1,cv2.LINE_AA);cv2.circle(image,tuple(p),8,(245,245,245),2,cv2.LINE_AA)

def main():
    p=argparse.ArgumentParser();p.add_argument("csv",type=Path);p.add_argument("--kick-a",required=True);p.add_argument("--kick-b",required=True);p.add_argument("--camera",default="C");p.add_argument("--camera-a");p.add_argument("--camera-b");p.add_argument("--canonical-foot",choices=("L","R"));p.add_argument("--output",type=Path,required=True);p.add_argument("--fps",type=float,default=10);p.add_argument("--size",type=int,default=720);a=p.parse_args()
    camera_a=a.camera_a or a.camera;camera_b=a.camera_b or a.camera
    rows_a,touch_a,track_a,foot_a=load(a.csv,a.kick_a,camera_a);rows_b,touch_b,track_b,foot_b=load(a.csv,a.kick_b,camera_b)
    mirror_a=bool(a.canonical_foot and foot_a!=a.canonical_foot);mirror_b=bool(a.canonical_foot and foot_b!=a.canonical_foot)
    # Keep original frame cadence and align the two contact frames. Earlier
    # unmatched frames are rendered with only the longer sequence visible.
    before=max(touch_a,touch_b);after=max(len(rows_a)-touch_a-1,len(rows_b)-touch_b-1);total=before+1+after
    offset_a=before-touch_a;offset_b=before-touch_b
    scale_a=(a.size*.68)/float(np.median([float(r["bbox_h"]) for r in rows_a]));scale_b=(a.size*.68)/float(np.median([float(r["bbox_h"]) for r in rows_b]))
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_name(a.output.stem+"_mpeg4.mp4")
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*"mp4v"),a.fps,(a.size,a.size))
    for t in range(total):
        im=np.full((a.size,a.size,3),(18,20,27),dtype=np.uint8)
        ia=t-offset_a;ib=t-offset_b
        if 0<=ia<len(rows_a):draw(im,points(rows_a[ia],a.size,scale_a,mirror_a),(255,210,50))
        if 0<=ib<len(rows_b):draw(im,points(rows_b[ib],a.size,scale_b,mirror_b),(220,70,255))
        label_a=f"kick {a.kick_a} / {camera_a}"+(" mirrored" if mirror_a else "")
        label_b=f"kick {a.kick_b} / {camera_b}"+(" mirrored" if mirror_b else "")
        cv2.putText(im,label_a,(25,40),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,210,50),2,cv2.LINE_AA)
        cv2.putText(im,label_b,(a.size-225,40),cv2.FONT_HERSHEY_SIMPLEX,.72,(220,70,255),2,cv2.LINE_AA)
        cv2.putText(im,"last_touch aligned",(a.size//2-120,a.size-28),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),2,cv2.LINE_AA)
        if t==before:
            cv2.rectangle(im,(15,58),(a.size-15,102),(30,30,180),-1);cv2.putText(im,"CONTACT FRAME",(a.size//2-95,89),cv2.FONT_HERSHEY_SIMPLEX,.78,(255,255,255),2,cv2.LINE_AA)
        writer.write(im)
    writer.release()
    subprocess.run(["/usr/bin/ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(temp),"-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(a.output)],check=True)
    meta={"kick_a":a.kick_a,"camera_a":camera_a,"track_a":track_a,"foot_a":foot_a,"mirrored_a":mirror_a,"frames_a":len(rows_a),"kick_b":a.kick_b,"camera_b":camera_b,"track_b":track_b,"foot_b":foot_b,"mirrored_b":mirror_b,"frames_b":len(rows_b),"canonical_foot":a.canonical_foot,"mirror_operation":"hip-centered x reflection plus COCO left/right joint swap","alignment":"last_touch","output_frames":total,"playback_fps":a.fps,"source_fps":"unknown"}
    a.output.with_suffix(".json").write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
