#!/usr/bin/env python3
"""Render the actual 20-sample, 8-joint canonical CSV without changing poses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)
EDGES = ((5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
INDEX = {j: i for i, j in enumerate(JOINTS)}
LEFT, RIGHT, TORSO = (185, 125, 20), (30, 135, 235), (115, 100, 95)  # BGR
BACKGROUND, TEXT, MUTED = (247, 247, 244), (48, 43, 39), (125, 117, 108)
FPS, HOLD, END_HOLD, REPEATS = 25, 5, 25, 3
ANGLE_COLUMNS = ("knee_angle_l_n", "knee_angle_r_n", "hip_angle_l_n", "hip_angle_r_n")


def label(im, value, xy, size=.5, color=TEXT, thick=1):
    cv2.putText(im, value, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def load(path):
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["kick_id"]].append(row)
    sequences = []
    for name, rows in groups.items():
        assert len(rows) == 20 and [int(r["sample_20_idx"]) for r in rows] == list(range(20))
        assert rows[-1]["last_touch"] == "1" and all(r["r_or_l"] == "R" for r in rows)
        poses = np.array([[[float(r[f"kp_{j}_xn"]), float(r[f"kp_{j}_yn"])] for j in JOINTS] for r in rows])
        assert np.isfinite(poses).all() and np.abs((poses[:, 2]+poses[:, 3])/2).max() < 1e-10
        sequences.append({"name": name, "rows": rows, "poses": poses})
    assert len(sequences) == 14
    all_points = np.concatenate([s["poses"].reshape(-1, 2) for s in sequences])
    return sequences, all_points.min(0)-.2, all_points.max(0)+.2


def transform(low, high, rect):
    x, y, w, h = rect
    scale = float(np.min(np.array([w, h])/(high-low)))
    anchor = np.array([x+w/2, y+h/2])-(low+high)/2*scale
    return scale, anchor


def edge_color(a, b):
    if (a, b) in ((5, 11), (11, 13), (13, 15)):
        return LEFT
    if (a, b) in ((6, 12), (12, 14), (14, 16)):
        return RIGHT
    return TORSO


def skeleton(im, pose, low, high, rect, numbered=False, thickness=2):
    scale, anchor = transform(low, high, rect)
    x, y, w, h = rect
    origin = tuple(np.rint(anchor).astype(int))
    cv2.line(im, (x, origin[1]), (x+w, origin[1]), (225, 222, 219), 1)
    cv2.line(im, (origin[0], y), (origin[0], y+h), (225, 222, 219), 1)
    points = np.rint(pose*scale+anchor).astype(int)
    assert ((points[:, 0] >= x) & (points[:, 0] <= x+w)).all()
    assert ((points[:, 1] >= y) & (points[:, 1] <= y+h)).all()
    lookup = {j: tuple(points[INDEX[j]]) for j in JOINTS}
    for a, b in EDGES:
        cv2.line(im, lookup[a], lookup[b], edge_color(a, b), thickness, cv2.LINE_AA)
    for j in JOINTS:
        color = LEFT if j in (5, 11, 13, 15) else RIGHT
        cv2.circle(im, lookup[j], max(3, thickness+1), color, -1, cv2.LINE_AA)
        if numbered:
            label(im, str(j), (lookup[j][0]+7, lookup[j][1]-5), .4, color)
    cv2.drawMarker(im, origin, (100, 100, 100), cv2.MARKER_CROSS, 9, 1)
    return scale


def grid_step(sequences, step, low, high):
    im = np.full((1440, 1920, 3), BACKGROUND, np.uint8)
    label(im, "14 penalty kicks | normalized 8-joint graphs", (26, 38), .95, thick=2)
    label(im, f"Sample {step+1:02d}/20  |  right-foot canonical  |  shared torso-unit scale  |  pelvis = (0, 0)", (26, 71), .6)
    for i, item in enumerate(sequences):
        x, y = (i % 4)*480, 95+(i//4)*310
        cv2.rectangle(im, (x+8, y+4), (x+472, y+304), (226, 222, 217), 1)
        row = item["rows"][step]
        label(im, item["name"].removesuffix(".mp4"), (x+22, y+28), .6, thick=2)
        label(im, f"view {row['camera_direction']} | foot {row['original_r_or_l']} -> R | dir {row['original_kick_direction']} -> {row['kick_direction']}", (x+22, y+52), .44)
        skeleton(im, item["poses"][step], low, high, (x+25, y+65, 425, 202))
        label(im, f"source frame {row['source_frame']}  |  t={float(row['time_sec']):.2f}s", (x+22, y+289), .43, MUTED)
        if step == 19:
            label(im, "CONTACT", (x+350, y+28), .43, (45, 80, 200), 2)
    x, y = 2*480+25, 95+3*310+32
    label(im, "ANATOMICAL LEFT", (x, y), .57, LEFT, 2)
    label(im, "ANATOMICAL RIGHT", (x, y+38), .57, RIGHT, 2)
    label(im, "Shoulder -> hip -> knee -> ankle", (x, y+80), .47)
    label(im, "Poses are the CSV's exact coordinates.", (x, y+125), .44, MUTED)
    label(im, "No new smoothing, interpolation or flip.", (x, y+154), .44, MUTED)
    label(im, "20 sampled steps aligned by progress.", (3*480+25, y), .47)
    label(im, "Each source frame/time is shown separately.", (3*480+25, y+37), .43, MUTED)
    label(im, "Image y-axis points downward.", (3*480+25, y+78), .46, MUTED)
    label(im, "Kicks 001 and 007 were excluded upstream.", (3*480+25, y+120), .41, MUTED)
    label(im, "Start -> contact | playback holds each sampled pose for inspection; it does not represent synchronized match time.", (26, 1378), .61)
    cv2.rectangle(im, (26, 1401), (1894, 1414), (220, 218, 214), -1)
    cv2.rectangle(im, (26, 1401), (26+round(1868*step/19), 1414), (140, 125, 110), -1)
    return im


def individual_step(item, step, low, high):
    im = np.full((900, 1280, 3), BACKGROUND, np.uint8)
    row = item["rows"][step]
    label(im, item["name"]+" | normalized joint graph", (25, 37), .9, thick=2)
    label(im, f"sample {step+1:02d}/20 | original frame {row['source_frame']} | time {float(row['time_sec']):.2f}s", (25, 74), .62)
    skeleton(im, item["poses"][step], low, high, (60, 116, 800, 685), numbered=True, thickness=4)
    label(im, "pelvis (0,0)", (65, 839), .52, MUTED)
    x = 910
    label(im, "8 JOINTS", (x, 145), .7, thick=2)
    for n, text in enumerate(("5 / 6  shoulders", "11 / 12  hips", "13 / 14  knees", "15 / 16  ankles")):
        label(im, text, (x, 185+35*n), .55)
    label(im, "LEFT", (x, 345), .65, LEFT, 2)
    label(im, "RIGHT", (x+145, 345), .65, RIGHT, 2)
    label(im, "JOINT ANGLES (degrees)", (x, 414), .5, thick=2)
    for k, (column, name) in enumerate(zip(ANGLE_COLUMNS, ("left knee", "right knee", "left hip", "right hip"))):
        label(im, f"{name}: {float(row[column])*180:.1f}", (x, 455+k*34), .56)
    label(im, f"Camera: {row['camera_direction']}", (x, 638), .52)
    label(im, f"Foot: {row['original_r_or_l']} -> R", (x, 676), .52)
    label(im, f"Direction: {row['original_kick_direction']} -> {row['kick_direction']}", (x, 714), .52)
    label(im, f"Torso scale: {float(row['clip_torso_scale']):.2f}px", (x, 752), .48, MUTED)
    label(im, "Exact sampled CSV poses | fixed x/y scale | y points down", (25, 883), .6)
    if step == 19:
        label(im, "CONTACT", (x, 828), .9, (45, 80, 200), 2)
    return im


def encode(images, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+"_temp.mp4")
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {temp}")
    for _ in range(REPEATS):
        for im in images:
            for _ in range(HOLD):
                writer.write(im)
        for _ in range(END_HOLD):
            writer.write(images[-1])
    writer.release()
    subprocess.run(["/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(temp),
                    "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    temp.unlink()
    stream = json.loads(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error", "-count_frames",
                "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames,width,height,codec_name",
                "-of", "json", str(path)]))["streams"][0]
    assert int(stream["nb_read_frames"]) == REPEATS*(20*HOLD+END_HOLD)
    assert (stream["width"], stream["height"], stream["codec_name"]) == (w, h, "h264")


def plot_pose(ax, pose, low, high, title, numbered=False):
    for a, b in EDGES:
        pts = pose[[INDEX[a], INDEX[b]]]
        color = np.array(edge_color(a, b)[::-1])/255
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2)
    for j in JOINTS:
        pt = pose[INDEX[j]]
        ax.scatter(pt[0], pt[1], s=16, color=np.array((LEFT if j in (5, 11, 13, 15) else RIGHT)[::-1])/255, zorder=3)
        if numbered:
            ax.annotate(str(j), pt, xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, linewidth=.6, color="#dddddd")
    ax.axvline(0, linewidth=.6, color="#dddddd")
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(high[1], low[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")


def static_sheets(sequences, low, high, output):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    for ax, item in zip(axes.flat, sequences):
        row = item["rows"][-1]
        title = (item["name"].removesuffix(".mp4")+f" | f={row['source_frame']}\n"
                 f"view {row['camera_direction']} | foot {row['original_r_or_l']} -> R | direction {row['original_kick_direction']} -> {row['kick_direction']}")
        plot_pose(ax, item["poses"][-1], low, high, title, numbered=True)
    for ax in list(axes.flat)[14:]:
        ax.axis("off")
    axes.flat[14].text(.05, .6, "LEFT: blue / RIGHT: orange\nShoulders, hips, knees, ankles\nPelvis origin; fixed torso-unit scale", transform=axes.flat[14].transAxes, fontsize=12)
    axes.flat[15].text(.05, .6, "Contact sample: 20/20\nExact preprocessed CSV coordinates\nImage coordinate y points down", transform=axes.flat[15].transAxes, fontsize=12)
    fig.suptitle("14 penalty kicks — normalized 8-joint graphs at contact", fontsize=18)
    fig.savefig(output/"all_14_contact_graphs.png", dpi=150)
    fig.savefig(output/"all_14_contact_graphs.pdf")
    plt.close(fig)
    for item in sequences:
        fig, axes = plt.subplots(4, 5, figsize=(16, 14), constrained_layout=True)
        for step, ax in enumerate(axes.flat):
            row = item["rows"][step]
            title = f"{step+1:02d}/20 | source f={row['source_frame']} | {float(row['time_sec']):.2f}s"
            if step == 19:
                title += "\nCONTACT"
            plot_pose(ax, item["poses"][step], low, high, title)
        fig.suptitle(item["name"]+" — right-foot canonical, fixed shared scale", fontsize=15)
        dest = output/item["name"].removesuffix(".mp4")
        dest.mkdir(exist_ok=True)
        fig.savefig(dest/"all_20_sample_graphs.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Main Data/penaltyfirst15_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/penaltyfirst15_joint_graphs"))
    a = parser.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {a.output_dir}")
    a.output_dir.mkdir(parents=True)
    source_hash = hashlib.sha256(a.input.read_bytes()).hexdigest()
    sequences, low, high = load(a.input)
    cv2.setNumThreads(2)
    images = [grid_step(sequences, step, low, high) for step in range(20)]
    print("Rendering all-14 animation", flush=True)
    encode(images, a.output_dir/"all_14_joint_graphs.mp4")
    cv2.imwrite(str(a.output_dir/"all_14_graphs_preview.png"), images[-1])
    gif_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(im, (960, 720)), cv2.COLOR_BGR2RGB)) for im in images]
    gif_frames[0].save(a.output_dir/"all_14_joint_graphs.gif", save_all=True, append_images=gif_frames[1:],
                       duration=[200]*19+[1200], loop=0, optimize=True)
    for item in sequences:
        print("Rendering", item["name"], flush=True)
        individual = [individual_step(item, step, low, high) for step in range(20)]
        dest = a.output_dir/item["name"].removesuffix(".mp4")
        dest.mkdir(exist_ok=True)
        encode(individual, dest/"joint_graph.mp4")
        cv2.imwrite(str(dest/"contact_graph.png"), individual[-1])
    print("Rendering static scientific figures", flush=True)
    static_sheets(sequences, low, high, a.output_dir)
    assert hashlib.sha256(a.input.read_bytes()).hexdigest() == source_hash
    metadata = {"input": str(a.input), "input_sha256": source_hash, "input_unchanged": True,
                "videos": 14, "csv_pose_frames": 280, "joints": list(JOINTS), "edges": list(EDGES),
                "bounds": [low.tolist(), high.tolist()], "same_isotropic_scale_for_all_kicks": True,
                "csv_coordinates_used_verbatim": True, "extra_interpolation": False,
                "extra_smoothing": False, "extra_mirroring": False,
                "playback": "20 exact sampled poses held 0.2s each plus 1s contact hold, repeated 3 times",
                "timeline_alignment": "sample progress; not synchronized source physical time",
                "display_fps": FPS, "decoded_frames_per_mp4": REPEATS*(20*HOLD+END_HOLD),
                "duration_per_mp4_sec": REPEATS*(20*HOLD+END_HOLD)/FPS,
                "all_15_mp4_files_fully_decoded": True,
                "source_frames": {s["name"]: [int(r["source_frame"]) for r in s["rows"]] for s in sequences}}
    (a.output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))
    lines = ["# 전처리 관절 그래프 시각화", "",
             "[전체 14개 동작 영상](all_14_joint_graphs.mp4) · [GIF](all_14_joint_graphs.gif)", "",
             "[접촉 프레임 비교 이미지](all_14_contact_graphs.png) · [PDF](all_14_contact_graphs.pdf)", "",
             "최종 CSV의 8관절 정규화 좌표를 그대로 사용했습니다. 골반 중심은 (0,0)이며 모든 영상의 화면 축척이 같습니다.",
             "파란색은 신체 왼쪽, 주황색은 신체 오른쪽입니다. y축은 원본 이미지처럼 아래로 증가합니다.",
             "오른발 기준으로 통일한 결과이고 카메라 시점은 원래 값을 유지합니다.",
             "20개 샘플의 진행률을 맞춰 표시하며 각 카드의 원본 프레임·시간은 서로 다릅니다.",
             "관절 좌표를 추가 보간하거나 매끄럽게 만들지 않았습니다. 각 샘플을 0.2초 유지하고 마지막 접촉 포즈를 추가 1초 유지합니다.",
             "동영상은 15초이며 3회 반복합니다. 개별 영상에는 관절 번호와 CSV 각도의 도 단위 표시도 포함합니다.", "",
             "001과 007은 승인된 전처리 규칙으로 최종 CSV에서 제외되어 여기에도 포함되지 않습니다.", "",
             "| PK | 개별 동작 영상 | 20프레임 그래프 이미지 |", "|---|---|---|"]
    for s in sequences:
        stem = s["name"].removesuffix(".mp4")
        lines.append(f"| {stem} | [영상]({stem}/joint_graph.mp4) | [20프레임]({stem}/all_20_sample_graphs.png) |")
    lines += ["", "재실행: 프로젝트 루트에서 `python visualize_penaltyfirst15_graphs.py --output-dir 새_출력폴더`", "",
              "[검증·입력 프레임 대응 정보](metadata.json)"]
    (a.output_dir/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output_dir": str(a.output_dir), "videos": 14, "frames": 280, "input_unchanged": True}))


if __name__ == "__main__":
    main()
