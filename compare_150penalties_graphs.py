#!/usr/bin/env python3
"""Audit pose compatibility and overlay new kicks with EPL, Women and CLT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visualize_penaltyfirst15_graphs import JOINTS, EDGES, INDEX, encode, label, transform

NEW = Path('Main Data/150penalties_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv')
OLD = Path('Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv')
DATASETS = ('EPL', 'Women', 'CLT')
COLORS = {'150penalties': (30, 135, 235), 'EPL': (210, 130, 25),
          'Women': (170, 80, 165), 'CLT': (95, 155, 35)}  # BGR, color per dataset
ANGLES = ('knee_angle_l_n', 'knee_angle_r_n', 'hip_angle_l_n', 'hip_angle_r_n')
FEATURES = [f'kp_{j}_{a}' for j in JOINTS for a in ('xn', 'yn')] + list(ANGLES)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, mirrored):
    groups = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        assert set(FEATURES) <= set(reader.fieldnames)
        for row in reader:
            groups[(row['dataset'], row['kick_id'])].append(row)
    sequences = []
    for key, rows in sorted(groups.items()):
        assert len(rows) == 20
        assert [int(r['sample_20_idx']) for r in rows] == list(range(20))
        assert all(r['r_or_l'] == 'R' for r in rows)
        assert [int(float(r['last_touch'])) for r in rows] == [0]*19+[1]
        frames = [float(r['frame']) for r in rows]
        assert all(a < b for a, b in zip(frames, frames[1:]))
        pose = np.array([[[float(r[f'kp_{j}_xn']), float(r[f'kp_{j}_yn'])] for j in JOINTS] for r in rows])
        angles = np.array([[float(r[c]) for c in ANGLES] for r in rows])
        assert np.isfinite(pose).all() and np.isfinite(angles).all()
        assert ((angles >= 0) & (angles <= 1)).all()
        pelvis = np.array([[float(r['pelvis_xc']), float(r['pelvis_yc'])] for r in rows])
        centered = np.array([[[float(r[f'kp_{j}_xc']), float(r[f'kp_{j}_yc'])] for j in JOINTS] for r in rows])
        scale = np.array([float(r['clip_torso_scale']) for r in rows])
        assert scale.min() > 0 and np.ptp(scale) < 1e-10
        normalization_error = float(abs((centered-pelvis[:, None])/scale[:, None, None]-pose).max())
        pelvis_error = float(abs((pose[:, 2]+pose[:, 3])/2).max())
        assert normalization_error < 1e-8 and pelvis_error < 1e-10
        angle_errors = []
        for col, triplet in enumerate(((11,13,15), (12,14,16), (5,11,13), (6,12,14))):
            a, b, c = (pose[:, INDEX[j]] for j in triplet)
            u, v = a-b, c-b
            value = np.arccos(np.clip(np.sum(u*v, axis=1)/(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1)), -1, 1))/np.pi
            angle_errors.append(float(abs(value-angles[:, col]).max()))
        assert max(angle_errors) < 1e-6
        r = rows[0]
        foot = r.get('original_r_or_l') or ('L' if key in mirrored else 'R')
        sequences.append({'dataset': key[0], 'kick_id': key[1], 'camera': r['camera_direction'],
                          'original_foot': foot, 'direction': r['kick_direction'], 'poses': pose,
                          'angles': angles, 'frames': frames, 'rows': rows,
                          'normalization_error': normalization_error, 'pelvis_error': pelvis_error,
                          'angle_error': max(angle_errors)})
    return sequences


def select_pair(current, references, dataset, rng):
    same_direction = [s for s in references if s['dataset'] == dataset and s['direction'] == current['direction']]
    rules = [('camera+original_foot+direction', lambda s: s['camera'] == current['camera'] and s['original_foot'] == current['original_foot']),
             ('camera+direction', lambda s: s['camera'] == current['camera']),
             ('original_foot+direction', lambda s: s['original_foot'] == current['original_foot']),
             ('direction_only', lambda s: True)]
    for name, test in rules:
        pool = [s for s in same_direction if test(s)]
        if pool:
            chosen = rng.choice(pool)
            return chosen, name, len(pool)
    return None, 'no_same_direction_reference', 0


def bounds(sequences):
    points = np.concatenate([s['poses'].reshape(-1, 2) for s in sequences])
    return points.min(0)-.18, points.max(0)+.18


def draw_skeleton(im, pose, low, high, rect, color, width=2):
    scale, anchor = transform(low, high, rect)
    x, y, w, h = rect
    origin = tuple(np.rint(anchor).astype(int))
    cv2.line(im, (x, origin[1]), (x+w, origin[1]), (225, 225, 225), 1)
    cv2.line(im, (origin[0], y), (origin[0], y+h), (225, 225, 225), 1)
    points = np.rint(pose*scale+anchor).astype(int)
    assert ((points[:, 0] >= x) & (points[:, 0] <= x+w)).all()
    assert ((points[:, 1] >= y) & (points[:, 1] <= y+h)).all()
    layer = im.copy()
    for a, b in EDGES:
        cv2.line(layer, tuple(points[INDEX[a]]), tuple(points[INDEX[b]]), color, width, cv2.LINE_AA)
    for p in points:
        cv2.circle(layer, tuple(p), 3, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(layer, .83, im, .17, 0)


def render(current, pairs, dataset, step, low, high):
    im = np.full((1440, 1920, 3), (250, 250, 250), np.uint8)
    label(im, f'150penalties vs {dataset} | normalized 8-joint OVERLAP', (25, 38), .95, thick=2)
    label(im, f'Sample {step+1:02d}/20 | orange: 150penalties | reference: {dataset} | common torso-unit scale', (25, 74), .61)
    for i, new in enumerate(current):
        old, rule, n = pairs[new['kick_id']][dataset]
        x, y = i%4*480, 98+i//4*310
        cv2.rectangle(im, (x+8,y+2), (x+472,y+304), (225,225,225), 1)
        label(im, new['kick_id'].removesuffix('.mp4'), (x+20,y+28), .55, COLORS['150penalties'], 2)
        subtitle = f"{dataset}: {old['kick_id']}" if old else f'{dataset}: no same-direction kick'
        label(im, subtitle, (x+20,y+50), .43, COLORS[dataset])
        rect = (x+24,y+62,430,197)
        if old:
            im = draw_skeleton(im, old['poses'][step], low, high, rect, COLORS[dataset])
        im = draw_skeleton(im, new['poses'][step], low, high, rect, COLORS['150penalties'])
        if old:
            cams = f"view {new['camera']}/{old['camera'] or '?'}"
            foot = f"orig foot {new['original_foot']}/{old['original_foot']}"
            label(im, f"{cams} | {foot} | canonical dir {new['direction']}", (x+20,y+277), .37)
            label(im, f"f {new['frames'][step]:g}/{old['frames'][step]:g} | {rule}", (x+20,y+295), .35)
        else:
            label(im, 'Reference omitted: direction is unavailable.', (x+20,y+289), .41)
    label(im, 'NEW: 150penalties', (980,1075), .65, COLORS['150penalties'], 2)
    label(im, f'REFERENCE: {dataset}', (980,1122), .65, COLORS[dataset], 2)
    for n, text in enumerate(('Pelvis = (0,0); same x/y scale.', '20 exact CSV samples, start -> contact.', 'Sample progress alignment.', 'Camera / original foot shown for each pair.', 'References selected by labels, seed 42.', 'No pose-similarity based selection.')):
        label(im, text, (1460,1070+n*35), .40)
    label(im, 'Right-foot canonical | y points down | endpoint: contact | frames align by progress, source times differ', (25,1380), .6)
    return im


def contact_figure(current, pairs, dataset, low, high, output):
    fig, axes = plt.subplots(4,4,figsize=(16,16),constrained_layout=True)
    for ax, new in zip(axes.flat,current):
        old, rule, n = pairs[new['kick_id']][dataset]
        for seq in [old,new]:
            if seq is None: continue
            color = np.array(COLORS[seq['dataset']][::-1])/255
            p=seq['poses'][-1]
            for a,b in EDGES:
                pts=p[[INDEX[a],INDEX[b]]]
                ax.plot(pts[:,0],pts[:,1],color=color,linewidth=2,alpha=.8)
            ax.scatter(p[:,0],p[:,1],s=13,color=color,alpha=.8)
        ax.axhline(0,color='#ddd',linewidth=.6);ax.axvline(0,color='#ddd',linewidth=.6)
        ax.set_xlim(low[0],high[0]);ax.set_ylim(high[1],low[1]);ax.set_aspect('equal');ax.tick_params(labelsize=6)
        ref=old['kick_id'] if old else 'no same-direction reference'
        ax.set_title(f"{new['kick_id']} / {ref}\n{rule}",fontsize=8)
    for ax in list(axes.flat)[14:]: ax.axis('off')
    axes.flat[14].text(.02,.6,f'Orange: 150penalties\nReference: {dataset}\nSame axis scale across all exports',fontsize=11)
    axes.flat[15].text(.02,.6,'Contact sample 20/20\nCoordinates unchanged\nLabel-based selection, seed 42',fontsize=11)
    fig.suptitle(f'150penalties vs {dataset} — contact pose overlap',fontsize=18)
    fig.savefig(output/f'vs_{dataset}_contact.png',dpi=150)
    fig.savefig(output/f'vs_{dataset}_contact.pdf')
    plt.close(fig)


HTML = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>데이터셋 관절 그래프 overlap</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f5f3;color:#242424}main{max-width:1320px;margin:auto}h1{font-size:25px}select,button{font:inherit;padding:7px;max-width:100%}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}.box{background:white;padding:12px;border-radius:8px;flex:1;min-width:235px}.box small{display:block;margin-top:8px;color:#555}canvas{background:#fff;width:100%;height:auto;border:1px solid #ddd}input[type=range]{width:60%}.toolbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:15px 0}#details{white-space:pre-wrap;line-height:1.7}.note{color:#555;font-size:14px}</style>
<main><h1>150penalties × EPL · Women · CLT 관절 그래프 overlap</h1>
<p>주황색: 150penalties / 파란색: EPL / 보라색: Women / 초록색: CLT. 골반 중심 (0,0), 몸통 길이 단위, 오른발 기준 8관절입니다.</p>
<div class="controls"><div class="box"><b>현재 데이터</b><p><select id="current"></select></p><button id="reset">조건에 맞춘 기본 표본</button><small>기본 표본은 포즈 유사도가 아니라 카메라·원래 킥 발·정규화된 킥 방향으로 선택했습니다.</small></div>
<div class="box" id="EPLbox"></div><div class="box" id="Womenbox"></div><div class="box" id="CLTbox"></div></div>
<div class="toolbar"><button id="play">재생</button><input id="step" type="range" min="0" max="19" value="19"><b id="frameLabel"></b><label><input id="global" type="checkbox">전체 377개 킥에 고정 축 적용</label></div>
<canvas id="plot" width="1280" height="880"></canvas><p id="details"></p>
<p class="note">20개 샘플의 진행률을 맞춥니다. 같은 샘플 번호라도 실제 시간·보폭은 다를 수 있습니다. CLT의 카메라 메타데이터는 없습니다. 기본 축 범위는 선택한 킥들의 20프레임 전체에서 고정되며, 좌표를 회전·반사·보간하거나 서로 크기를 맞추지 않습니다. 전체 고정 축은 큰 좌표값도 포함합니다.</p>
<p><a href="README.md">영상·비교 이미지·검증 보고서</a></p></main>
<script>const DATA=__DATA__;
const colors={150penalties:'#eb871e',EPL:'#1982d2',Women:'#a550aa',CLT:'#239b5f'};
const edges=[[0,1],[0,2],[1,3],[2,3],[2,4],[4,6],[3,5],[5,7]],names=['EPL','Women','CLT'];
const $=id=>document.getElementById(id),ctx=$('plot').getContext('2d');let timer=null,hold=0;
function option(select,value,text){const el=document.createElement('option');el.value=value;el.textContent=text;select.appendChild(el)}
DATA.current.forEach((s,i)=>option($('current'),i,s.kick_id));
names.forEach(ds=>{const box=$(ds+'box');const toggle=document.createElement('input');toggle.type='checkbox';toggle.id=ds+'show';toggle.checked=true;box.appendChild(toggle);const title=document.createElement('b');title.textContent=' '+ds;title.style.color=colors[ds];box.appendChild(title);const p=document.createElement('p'),select=document.createElement('select');select.id=ds;option(select,'','겹치지 않음');DATA.reference.forEach((s,i)=>{if(s.dataset===ds)option(select,i,s.kick_id+' | view '+(s.camera||'?')+' | foot '+s.original_foot+' | dir '+s.direction)});p.appendChild(select);box.appendChild(p);const note=document.createElement('small');note.id=ds+'note';box.appendChild(note);select.onchange=draw;toggle.onchange=draw});
function defaults(){const selected=DATA.matches[$('current').value];names.forEach(ds=>{$(ds).value=selected[ds].reference_index===null?'':selected[ds].reference_index});draw()}
function selected(){const result=[DATA.current[+$('current').value]];names.forEach(ds=>{if($(ds+'show').checked&&$(ds).value!=='')result.push(DATA.reference[+$(ds).value])});return result}
function range(seqs){let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];seqs.forEach(s=>s.poses.forEach(p=>p.forEach(q=>{for(let a=0;a<2;a++){lo[a]=Math.min(lo[a],q[a]);hi[a]=Math.max(hi[a],q[a])}})));return [lo.map(x=>x-.18),hi.map(x=>x+.18)]}
const fullBounds=range([...DATA.current,...DATA.reference]);
function draw(){const seqs=selected(),step=+$('step').value,cur=seqs[0];$('frameLabel').textContent=(step+1)+'/20'+(step===19?' · 공 접촉':'');ctx.clearRect(0,0,1280,880);const [lo,hi]=$('global').checked?fullBounds:range(seqs);const scale=Math.min(1080/(hi[0]-lo[0]),695/(hi[1]-lo[1]));const anchor=[640-(lo[0]+hi[0])/2*scale,435-(lo[1]+hi[1])/2*scale];const point=q=>[anchor[0]+q[0]*scale,anchor[1]+q[1]*scale];ctx.strokeStyle='#e5e5e5';ctx.lineWidth=1;ctx.fillStyle='#777';ctx.font='15px system-ui';for(let axis=0;axis<2;axis++){const tick=(hi[axis]-lo[axis])>6?1:.5;for(let v=Math.ceil(lo[axis]/tick)*tick;v<=hi[axis];v+=tick){const q=point(axis===0?[v,0]:[0,v]);ctx.beginPath();if(axis===0){ctx.moveTo(q[0],80);ctx.lineTo(q[0],792);ctx.fillText(v.toFixed(1),q[0]+3,816)}else{ctx.moveTo(90,q[1]);ctx.lineTo(1190,q[1]);ctx.fillText(v.toFixed(1),48,q[1]+5)}ctx.stroke()}}ctx.strokeStyle='#aaa';const zero=point([0,0]);ctx.beginPath();ctx.moveTo(90,zero[1]);ctx.lineTo(1190,zero[1]);ctx.moveTo(zero[0],80);ctx.lineTo(zero[0],792);ctx.stroke();
seqs.slice().reverse().forEach(s=>{const p=s.poses[step].map(point);ctx.globalAlpha=.82;ctx.strokeStyle=colors[s.dataset];ctx.fillStyle=colors[s.dataset];ctx.lineWidth=s.dataset==='150penalties'?4:3;edges.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(...p[a]);ctx.lineTo(...p[b]);ctx.stroke()});p.forEach(q=>{ctx.beginPath();ctx.arc(...q,5,0,2*Math.PI);ctx.fill()})});ctx.globalAlpha=1;ctx.font='bold 20px system-ui';ctx.fillStyle='#333';ctx.fillText(cur.kick_id+' · '+(step+1)+'/20',25,32);ctx.font='17px system-ui';seqs.forEach((s,i)=>{ctx.fillStyle=colors[s.dataset];ctx.fillText(s.dataset+' / '+s.kick_id,25+i*310,62)});ctx.fillStyle='#555';ctx.fillText('골반 (0,0) · 같은 x/y 축척 · y는 아래로 증가 · 접촉/가정 접촉은 20/20',25,855);
const lines=seqs.map(s=>{const a=s.angles[step].map(v=>(v*180).toFixed(1));return s.dataset+' / '+s.kick_id+' | 원본 f='+s.frames[step]+' | camera='+ (s.camera||'미상')+' | 원래 발='+s.original_foot+' | 정규화 방향='+s.direction+' | 무릎 L/R='+a[0]+'/'+a[1]+'°, 고관절 L/R='+a[2]+'/'+a[3]+'°'});$('details').textContent=lines.join('\n');names.forEach(ds=>{const val=$(ds).value;if(val===''){$(ds+'note').textContent='같은 방향의 기본 비교 표본이 없습니다. 다른 표본을 직접 선택할 수 있습니다.';return}const s=DATA.reference[+val];$(ds+'note').textContent='카메라 '+(cur.camera===s.camera?'일치':'다름/미상')+' · 원래 발 '+(cur.original_foot===s.original_foot?'일치':'다름')+' · 정규화 방향 '+(cur.direction===s.direction?'일치':'다름')})}
$('current').onchange=defaults;$('reset').onclick=defaults;$('step').oninput=()=>{hold=0;draw()};$('global').onchange=draw;
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='재생';return}$('play').textContent='일시정지';if(+$('step').value===19)$('step').value=0;draw();timer=setInterval(()=>{let n=+$('step').value;if(n===19&&hold++<5)return;hold=0;$('step').value=(n+1)%20;draw()},200)};defaults();
</script></html>'''


def payload(s):
    return {k:(s[k].tolist() if isinstance(s[k],np.ndarray) else s[k]) for k in ('dataset','kick_id','camera','original_foot','direction','poses','angles','frames')}


def write_focus(current, pairs, output):
    # The first center-view, originally right-footed, left-directed kick is
    # selected by metadata. Its references remain the seeded label selections.
    new=next(s for s in current if s['camera']=='C' and s['original_foot']=='R' and s['direction']=='L')
    refs=[pairs[new['kick_id']][ds][0] for ds in DATASETS]
    low,high=bounds([new]+[s for s in refs if s])
    images=[]
    for step in range(20):
        im=np.full((1000,1920,3),(250,250,250),np.uint8)
        label(im, f"{new['kick_id']} | three-dataset OVERLAP | sample {step+1:02d}/20",(25,38),.95,thick=2)
        label(im,'Orange: 150penalties | pelvis (0,0) | shared x/y scale | same 20 sampled progress steps',(25,76),.64)
        for i,(ds,old) in enumerate(zip(DATASETS,refs)):
            x=i*640
            label(im,f"{ds} / {old['kick_id'] if old else 'unavailable'}",(x+25,120),.68,COLORS[ds],2)
            rect=(x+60,145,520,625)
            if old: im=draw_skeleton(im,old['poses'][step],low,high,rect,COLORS[ds],4)
            im=draw_skeleton(im,new['poses'][step],low,high,rect,COLORS['150penalties'],4)
            if old:
                label(im,f"source frames: {new['frames'][step]:g} / {old['frames'][step]:g}",(x+25,818),.57)
                label(im,f"view {new['camera']} / {old['camera'] or '?'} | original foot {new['original_foot']} / {old['original_foot']}",(x+25,854),.55)
                label(im,f"canonical direction: {new['direction']} / {old['direction']}",(x+25,890),.55)
                label(im,pairs[new['kick_id']][ds][1],(x+25,926),.48)
        label(im,'CONTACT' if step==19 else 'RUN-UP -> CONTACT',(25,977),.65,thick=2)
        images.append(im)
    encode(images,output/'focused_three_dataset_overlap.mp4')
    assert cv2.imwrite(str(output/'focused_contact_overlap.png'),images[-1])
    gif=[Image.fromarray(cv2.cvtColor(cv2.resize(im,(1152,600)),cv2.COLOR_BGR2RGB)) for im in images]
    gif[0].save(output/'focused_three_dataset_overlap.gif',save_all=True,append_images=gif[1:],duration=[200]*19+[1200],loop=0)
    return {'new_kick':new['kick_id'],'selection':'first center-view, original R foot, canonical L direction; no pose similarity selection',
            'references':{ds:s['kick_id'] if s else None for ds,s in zip(DATASETS,refs)},
            'shared_bounds':[low.tolist(),high.tolist()],
            'all_coordinates_preserved':True,'all_video_frames_decoded':True}


def merge_csv(old_path,new_path,output):
    from preprocess_penaltyfirst15 import read_csv
    old_fields,old_rows=read_csv(old_path);new_fields,new_rows=read_csv(new_path)
    fields=old_fields+[c for c in new_fields if c not in old_fields]
    assert set(old_fields)<=set(new_fields)
    old_keys={(r['dataset'],r['kick_id']) for r in old_rows}
    new_keys={(r['dataset'],r['kick_id']) for r in new_rows}
    assert not old_keys&new_keys
    # The previously added first15 clips occur again in the new complete source set.
    previous=Path('Main Data/penaltyfirst15_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv')
    _,previous_rows=read_csv(previous)
    previous_names={r['kick_id'] for r in previous_rows}
    assert previous_names<={r['kick_id'] for r in new_rows}
    merged=[{c:r.get(c,'') for c in fields} for r in old_rows+new_rows]
    with output.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(merged)
    saved_fields,saved=read_csv(output)
    assert fields==saved_fields and saved==merged
    for original,actual in zip(old_rows+new_rows,saved):
        assert all(actual[c]==value for c,value in original.items())
    sample_keys=[(r['dataset'],r['kick_id'],r['sample_20_idx']) for r in saved]
    assert len(sample_keys)==len(set(sample_keys))
    sequences=defaultdict(list)
    for r in saved:sequences[(r['dataset'],r['kick_id'])].append(r)
    for rows in sequences.values():
        assert len(rows)==20 and [int(r['sample_20_idx']) for r in rows]==list(range(20))
        assert [int(float(r['last_touch'])) for r in rows]==[0]*19+[1]
        assert all(r['r_or_l']=='R' and all(np.isfinite(float(r[c])) for c in FEATURES) for r in rows)
    report={'output':str(output),'output_sha256':digest(output),'input_sha256':{str(p):digest(p) for p in (old_path,new_path)},
            'total_rows':len(saved),'total_kicks':len(sequences),'rows_per_kick':20,'columns':len(fields),
            'datasets':{ds:{'kicks':sum(k[0]==ds for k in sequences),'rows':sum(r['dataset']==ds for r in saved)} for ds in sorted({k[0] for k in sequences})},
            'extra_columns':fields[len(old_fields):],'row_order':'Original EPL/CLT/Women combined CSV followed by 150penalties.',
            'absent_provenance_values':'Empty for old rows; never inferred.',
            'first15_overlap':{'overlapping_videos':len(previous_names),'clip_names':sorted(previous_names),
                               'policy':'The old base excludes first15 additions. Those clips are included once via the new 150penalties input.'},
            'model_feature_columns':FEATURES,'validation':{'passed':True,'source_cell_values_preserved':True,'unique_sample_keys':True,
            'all_sequences_20_frames':True,'contact_is_last_sample':True,'all_model_features_finite':True,'all_canonical_feet_R':True}}
    output.with_suffix('.merge_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report

def representative_subset(current,count=14):
    pool=[s for s in current if int(s['kick_id'][:3])>15]
    selected=[];seen=set()
    for s in pool:
        key=(s['camera'],s['original_foot'],s['direction'])
        if key not in seen:selected.append(s);seen.add(key)
    for s in pool:
        if len(selected)>=count:break
        if all(s is not v for v in selected):selected.append(s)
    return selected[:count]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('runs/150penalties_cross_dataset_overlap'))
    parser.add_argument('--merged-output',type=Path,default=Path('Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled_with_150penalties.csv'))
    a=parser.parse_args()
    if a.output_dir.exists() or a.merged_output.exists():raise FileExistsError('Output already exists')
    audit_path=Path('runs/main_data_right_canonical/mirrored_rows.csv')
    full_paths=[Path('Main Data/Combined_penalty_pose_keypoints_normalized.csv'),Path('Main Data/150penalties_pose_keypoints_normalized.csv')]
    source_paths=[NEW,OLD,audit_path,*full_paths]
    hashes={str(p):digest(p) for p in source_paths}
    with audit_path.open() as f:mirrored={(r['dataset'],r['kick_id']) for r in csv.DictReader(f)}
    current,references=load(NEW,mirrored),load(OLD,mirrored)
    assert len(current)>0 and len(references)==226
    all_sequences=current+references
    full_lengths=defaultdict(list)
    for p in full_paths:
        with p.open() as f:
            for r in csv.DictReader(f):
                try:length=float(r['torso_length'])
                except ValueError:continue
                if np.isfinite(length) and length>0:full_lengths[(r['dataset'],r['kick_id'])].append(length)
    scale_errors=[abs(np.median(full_lengths[(s['dataset'],s['kick_id'])])-float(s['rows'][0]['clip_torso_scale'])) for s in all_sequences]
    assert max(scale_errors)<1e-8
    a.output_dir.mkdir(parents=True)
    a.merged_output.parent.mkdir(parents=True,exist_ok=True)
    merge=merge_csv(OLD,NEW,a.merged_output)
    pairs={};rows=[];rng=random.Random(42)
    for new in current:
        pairs[new['kick_id']]={}
        for ds in DATASETS:
            old,rule,n=select_pair(new,references,ds,rng);pairs[new['kick_id']][ds]=(old,rule,n)
            distance=float(np.sqrt(np.mean(np.sum((new['poses']-old['poses'])**2,axis=2)))) if old else None
            rows.append({'new_kick':new['kick_id'],'reference_dataset':ds,'reference_kick':old['kick_id'] if old else '',
                         'selection_rule':rule,'candidate_count':n,'new_camera':new['camera'],'reference_camera':old['camera'] if old else '',
                         'new_original_foot':new['original_foot'],'reference_original_foot':old['original_foot'] if old else '',
                         'canonical_direction':new['direction'],'rms_joint_distance_torso_units':distance})
    with (a.output_dir/'matched_pairs.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    statistics={}
    for ds in sorted({s['dataset'] for s in all_sequences}):
        seq=[s for s in all_sequences if s['dataset']==ds];p=np.concatenate([s['poses'] for s in seq])
        statistics[ds]={'kicks':len(seq),'rows':len(p),'bounds':[p.min((0,1)).tolist(),p.max((0,1)).tolist()],
                        'absolute_coordinate_percentiles':dict(zip(('p50','p95','p99','max'),np.percentile(abs(p),[50,95,99,100]).tolist()))}
    report={'input_sha256':hashes,'input_files_unchanged':True,'representation_compatible':True,'merged_output':str(a.merged_output),
            'joints':list(JOINTS),'edges':list(EDGES),'feature_order':FEATURES,'feature_shape_per_kick':[20,20],
            'validation':{'passed':True,'kicks':len(all_sequences),'rows':sum(len(s['rows']) for s in all_sequences),
                          'all_model_features_finite':True,'all_canonical_feet_R':True,'sample_indices_and_contact_endpoints_valid':True,
                          'max_pelvis_origin_error':max(s['pelvis_error'] for s in all_sequences),
                          'max_normalization_formula_error':max(s['normalization_error'] for s in all_sequences),
                          'max_full_sequence_median_scale_error':float(max(scale_errors)),
                          'max_angle_formula_error':max(s['angle_error'] for s in all_sequences)},
            'dataset_statistics':statistics,'selection_seed':42,'selection_based_on_pose_similarity':False,
            'selection_priority':['camera+original_foot+direction','camera+direction','original_foot+direction','direction_only'],
            'unavailable_same_direction_pairs':sum(not r['reference_kick'] for r in rows),'pairs':rows,
            'alignment':'20 ordered sample steps, including contact; source clock times and gait phases differ.',
            'coordinate_transforms_added':False,'interpretation':'Graph/feature representation is compatible. An overlap does not establish identical motions, distributions or model accuracy.',
            'CLT_camera':'Missing in source, not inferred.','HTML':'All 151 new and all 226 reference clips; offline selection and timeline controls.'}
    matches=[]
    for new in current:
        entry={}
        for ds in DATASETS:
            old,rule,n=pairs[new['kick_id']][ds]
            entry[ds]={'reference_index':next((i for i,s in enumerate(references) if old is s),None),'selection_rule':rule}
        matches.append(entry)
    web={'current':[payload(s) for s in current],'reference':[payload(s) for s in references],'matches':matches}
    html=HTML.replace("lines.join('\n')","lines.join('\\n')")
    # Dataset names beginning with a number require a quoted JS object key.
    html=html.replace("const colors={150penalties:","const colors={'150penalties':")
    (a.output_dir/'index.html').write_text(html.replace('__DATA__',json.dumps(web,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    displayed=representative_subset(current)
    low,high=bounds(displayed+[pairs[s['kick_id']][ds][0] for s in displayed for ds in DATASETS if pairs[s['kick_id']][ds][0]])
    report['static_example_clips']=[s['kick_id'] for s in displayed]
    report['static_example_selection']='First clip for each camera/original foot/canonical direction combination, then fill up to 14; IDs >15; no pose similarity ranking.'
    report['export_shared_bounds']=[low.tolist(),high.tolist()]
    cv2.setNumThreads(2)
    for ds in DATASETS:
        print('Rendering overlap:',ds,flush=True)
        images=[render(displayed,pairs,ds,i,low,high) for i in range(20)]
        encode(images,a.output_dir/f'vs_{ds}_overlap.mp4')
        cv2.imwrite(str(a.output_dir/f'vs_{ds}_contact_preview.png'),images[-1])
        contact_figure(displayed,pairs,ds,low,high,a.output_dir)
    print('Rendering enlarged comparison',flush=True)
    report['focused_example']=write_focus([s for s in current if int(s['kick_id'][:3])>15],pairs,a.output_dir)
    assert hashes=={str(p):digest(p) for p in source_paths}
    merge['validation']['source_files_unchanged']=True
    a.merged_output.with_suffix('.merge_report.json').write_text(json.dumps(merge,indent=2,ensure_ascii=False))
    report['all_4_mp4_files_fully_decoded']=True
    (a.output_dir/'compatibility_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    lines=['# 150penalties와 기존 관절 그래프 통합·오버레이','',
           '[151개 새 영상과 226개 기존 영상 직접 선택](index.html) · [세 데이터셋 확대 오버레이](focused_three_dataset_overlap.mp4) · [접촉 이미지](focused_contact_overlap.png)','',
           '주황: 150penalties / 파랑: EPL / 보라: Women / 초록: CLT. HTML 파일은 브라우저에서 열어 영상을 선택하고, 20프레임 슬라이더·재생 버튼·표시 체크박스로 직접 겹쳐 볼 수 있습니다. 네트워크 연결이 필요 없습니다.','',
           f'통합 CSV: `{a.merged_output}`. 기존 226개와 새 151개, 총 377개 영상·7,540행·115열입니다. 영상별 모델 입력은 20프레임 × (정규화 관절 좌표 16개 + 각도 4개)입니다. 첫15의 이전 14개 표본은 새 데이터에 모두 있으므로 이번 통합에서는 한 번씩 포함됩니다. 각 입력의 값과 행 순서는 보존했고, 기존에 없던 출처·신뢰도 열만 빈칸으로 확장했습니다.','',
           '## 호환성 검증','',
           '377개 영상의 7,540행을 검사했습니다. 관절 순서와 연결, 오른발 기준, 골반 원점, 전체 유지 구간 몸통 길이 중앙값으로 양축 정규화, 각도/π 정의, 시작·접촉 포함 20프레임과 유한한 모델 특징이 일치합니다. 기존 CSV와 새 CSV는 수정하지 않았습니다.','',
           f"최대 골반 원점 오차 {report['validation']['max_pelvis_origin_error']:.3g}, 정규화 수식 오차 {report['validation']['max_normalization_formula_error']:.3g}, 몸통 중앙값 오차 {report['validation']['max_full_sequence_median_scale_error']:.3g}, 각도 수식 오차 {report['validation']['max_angle_formula_error']:.3g}입니다.",'',
           '그래프와 특징 표현은 호환됩니다. 같은 프레임 번호를 겹쳤다고 실제 시간·보폭이 같아지는 것은 아니며, 선수 동작·카메라 투영·분포·모델 성능은 별도로 평가할 내용입니다.','',
           '## 표본과 축척','',
           '기본 비교 표본은 카메라+원래 발+정규화 방향을 우선으로 seed 42에서 선택했습니다. 조건이 부족하면 카메라+방향, 원래 발+방향, 방향만 일치하는 순서로 선택합니다. 포즈 유사도로 표본을 고르지 않았습니다. 모든 표본의 선택 이유는 matched_pairs.csv에 있습니다.','',
           'CLT는 카메라 정보가 없고, 정규화된 C 방향 표본도 없습니다. 새 C 방향 7개에는 CLT 기본 표본을 생략했습니다. HTML에서 다른 영상을 직접 선택할 수 있습니다.','',
           '내보낸 14개 표본의 이미지·영상은 동일한 축척을 사용합니다. 확대 예시는 선택된 네 영상의 전체 20프레임 범위에서 공통 축척을 계산합니다. HTML의 기본 축도 선택한 영상들의 전체 구간에서 고정되고, 전체 377개 고정 축으로 바꿀 수 있습니다. 원래 CSV 좌표를 추가 반전·회전·보간하거나 골격마다 크기를 따로 맞추지 않았습니다.','',
           '| 비교 | 14개 표본 오버레이 영상 | 접촉 그래프 |','|---|---|---|']
    for ds in DATASETS:lines.append(f'| {ds} | [MP4](vs_{ds}_overlap.mp4) | [PNG](vs_{ds}_contact.png) · [PDF](vs_{ds}_contact.pdf) |')
    lines+=['','[검증 보고서](compatibility_report.json) · [표본 대응표](matched_pairs.csv)','',
            '재실행: `python compare_150penalties_graphs.py --output-dir 새_폴더 --merged-output 새_통합파일.csv`']
    (a.output_dir/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'merged_output':str(a.merged_output),'representation_compatible':True,'videos':len(all_sequences),'rows':merge['total_rows'],'pairs':len(rows)},ensure_ascii=False))

if __name__=='__main__':main()
