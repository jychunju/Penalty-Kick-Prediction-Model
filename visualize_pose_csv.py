#!/usr/bin/env python3
"""Render a COCO kp_5..kp_16 sequence from CSV as a pose-only video."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


JOINTS = tuple(range(5, 17))
EDGES = ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
LEFT = {5,7,9,11,13,15}
RIGHT = {6,8,10,12,14,16}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("csv",type=Path)
    p.add_argument("--kick-id",required=True)
    p.add_argument("--camera-direction")
    p.add_argument("--track-id")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fps",type=float,default=10.0,help="visualization playback FPS; source FPS may be unknown")
    p.add_argument("--size",type=int,default=720)
    args=p.parse_args()

    with args.csv.open(newline="",encoding="utf-8-sig") as f:
        rows=[{k:(v or "").strip() for k,v in r.items()} for r in csv.DictReader(f) if (r.get("kick_id") or "").strip()==args.kick_id]
    if args.camera_direction is not None:
        rows=[r for r in rows if r.get("camera_direction")==args.camera_direction]
    if args.track_id is not None:
        rows=[r for r in rows if r.get("track_id")==args.track_id]
    if not rows: raise ValueError(f"kick_id {args.kick_id!r} not found")
    rows.sort(key=lambda r:int(float(r["frame"])))
    groups={(r.get("camera_direction",""),r.get("track_id","")) for r in rows}
    if len(groups)!=1: raise ValueError(f"kick_id has multiple camera/track groups: {sorted(groups)}")
    scales=np.asarray([float(r["bbox_h"]) for r in rows],dtype=float)
    median_bbox_h=float(np.median(scales))
    render_scale=(args.size*.68)/median_bbox_h
    hip_anchor=np.asarray([args.size*.5,args.size*.42])

    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_name(args.output.stem+"_mpeg4.mp4")
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*"mp4v"),args.fps,(args.size,args.size))
    if not writer.isOpened(): raise RuntimeError(f"cannot open writer: {temp}")
    missing_rows=0
    for row in rows:
        image=np.full((args.size,args.size,3),(18,20,27),dtype=np.uint8)
        points={}
        for joint in JOINTS:
            try:
                x=float(row[f"kp_{joint}_x"]);y=float(row[f"kp_{joint}_y"])
                if np.isfinite((x,y)).all():points[joint]=np.asarray([x,y])
            except ValueError: pass
        if len(points)<len(JOINTS):missing_rows+=1
        if 11 in points and 12 in points:
            hip_center=(points[11]+points[12])/2
        else:
            hip_center=np.asarray([float(row["bbox_cx"]),float(row["bbox_cy"])])
        canvas_points={j:np.rint((xy-hip_center)*render_scale+hip_anchor).astype(int) for j,xy in points.items()}
        for a,b in EDGES:
            if a in canvas_points and b in canvas_points:
                color=(60,210,255) if a in LEFT and b in LEFT else (255,155,70) if a in RIGHT and b in RIGHT else (180,180,190)
                cv2.line(image,tuple(canvas_points[a]),tuple(canvas_points[b]),color,5,cv2.LINE_AA)
        for joint,point in canvas_points.items():
            color=(60,210,255) if joint in LEFT else (255,155,70)
            cv2.circle(image,tuple(point),9,color,-1,cv2.LINE_AA)
            cv2.circle(image,tuple(point),9,(245,245,245),2,cv2.LINE_AA)
        frame=int(float(row["frame"])); touch=row.get("last_touch")=="1"
        cv2.putText(image,f"kick_id={args.kick_id}  frame={frame}",(24,40),cv2.FONT_HERSHEY_SIMPLEX,.85,(240,240,240),2,cv2.LINE_AA)
        cv2.putText(image,"LEFT",(24,args.size-28),cv2.FONT_HERSHEY_SIMPLEX,.7,(60,210,255),2,cv2.LINE_AA)
        cv2.putText(image,"RIGHT",(args.size-105,args.size-28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,155,70),2,cv2.LINE_AA)
        if touch:
            cv2.rectangle(image,(15,58),(args.size-15,105),(30,30,180),-1)
            cv2.putText(image,"LAST TOUCH / CONTACT FRAME",(args.size//2-215,91),cv2.FONT_HERSHEY_SIMPLEX,.78,(255,255,255),2,cv2.LINE_AA)
        writer.write(image)
    writer.release()
    ffmpeg=Path("/usr/bin/ffmpeg")
    if not ffmpeg.exists(): raise RuntimeError("/usr/bin/ffmpeg is required for H.264 output")
    subprocess.run([str(ffmpeg),"-hide_banner","-loglevel","error","-y","-i",str(temp),"-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)],check=True)
    metadata={"csv":str(args.csv),"kick_id":args.kick_id,"camera_direction":rows[0].get("camera_direction"),"track_id":rows[0].get("track_id"),"first_frame":int(float(rows[0]["frame"])),"last_frame":int(float(rows[-1]["frame"])),"frames":len(rows),"last_touch_frames":[int(float(r["frame"])) for r in rows if r.get("last_touch")=="1"],"rows_with_missing_joints":missing_rows,"median_bbox_h":median_bbox_h,"rendering":"framewise hip-centered; one fixed sequence scale based on median bbox_h","playback_fps":args.fps,"source_fps":"unknown"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps({**metadata,"temporary_video":str(temp),"output":str(args.output)},indent=2))


if __name__=="__main__":main()
