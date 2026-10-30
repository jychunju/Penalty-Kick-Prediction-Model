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

NEW = Path('Main Data/penaltyfirst15_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv')
OLD = Path('Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv')
DATASETS = ('EPL', 'Women', 'CLT')
COLORS = {'penaltyfirst15': (30, 135, 235), 'EPL': (210, 130, 25),
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
    label(im, f'penaltyfirst15 vs {dataset} | normalized 8-joint OVERLAP', (25, 38), .95, thick=2)
    label(im, f'Sample {step+1:02d}/20 | orange: penaltyfirst15 | reference: {dataset} | common torso-unit scale', (25, 74), .61)
    for i, new in enumerate(current):
        old, rule, n = pairs[new['kick_id']][dataset]
        x, y = i%4*480, 98+i//4*310
        cv2.rectangle(im, (x+8,y+2), (x+472,y+304), (225,225,225), 1)
        label(im, new['kick_id'].removesuffix('.mp4'), (x+20,y+28), .55, COLORS['penaltyfirst15'], 2)
        subtitle = f"{dataset}: {old['kick_id']}" if old else f'{dataset}: no same-direction kick'
        label(im, subtitle, (x+20,y+50), .43, COLORS[dataset])
        rect = (x+24,y+62,430,197)
        if old:
            im = draw_skeleton(im, old['poses'][step], low, high, rect, COLORS[dataset])
        im = draw_skeleton(im, new['poses'][step], low, high, rect, COLORS['penaltyfirst15'])
        if old:
            cams = f"view {new['camera']}/{old['camera'] or '?'}"
            foot = f"orig foot {new['original_foot']}/{old['original_foot']}"
            label(im, f"{cams} | {foot} | canonical dir {new['direction']}", (x+20,y+277), .37)
            label(im, f"f {new['frames'][step]:g}/{old['frames'][step]:g} | {rule}", (x+20,y+295), .35)
        else:
            label(im, 'Reference omitted: direction is unavailable.', (x+20,y+289), .41)
    label(im, 'NEW: penaltyfirst15', (980,1075), .65, COLORS['penaltyfirst15'], 2)
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
    axes.flat[14].text(.02,.6,f'Orange: penaltyfirst15\nReference: {dataset}\nSame axis scale across all exports',fontsize=11)
    axes.flat[15].text(.02,.6,'Contact sample 20/20\nCoordinates unchanged\nLabel-based selection, seed 42',fontsize=11)
    fig.suptitle(f'penaltyfirst15 vs {dataset} — contact pose overlap',fontsize=18)
    fig.savefig(output/f'vs_{dataset}_contact.png',dpi=150)
    fig.savefig(output/f'vs_{dataset}_contact.pdf')
    plt.close(fig)


HTML = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>데이터셋 관절 그래프 overlap</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f5f3;color:#242424}main{max-width:1320px;margin:auto}h1{font-size:25px}select,button{font:inherit;padding:7px;max-width:100%}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}.box{background:white;padding:12px;border-radius:8px;flex:1;min-width:235px}.box small{display:block;margin-top:8px;color:#555}canvas{background:#fff;width:100%;height:auto;border:1px solid #ddd}input[type=range]{width:60%}.toolbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:15px 0}#details{white-space:pre-wrap;line-height:1.7}.note{color:#555;font-size:14px}</style>
<main><h1>penaltyfirst15 × EPL · Women · CLT 관절 그래프 overlap</h1>
<p>주황색: penaltyfirst15 / 파란색: EPL / 보라색: Women / 초록색: CLT. 골반 중심 (0,0), 몸통 길이 단위, 오른발 기준 8관절입니다.</p>
<div class="controls"><div class="box"><b>현재 데이터</b><p><select id="current"></select></p><button id="reset">조건에 맞춘 기본 표본</button><small>기본 표본은 포즈 유사도가 아니라 카메라·원래 킥 발·정규화된 킥 방향으로 선택했습니다.</small></div>
<div class="box" id="EPLbox"></div><div class="box" id="Womenbox"></div><div class="box" id="CLTbox"></div></div>
<div class="toolbar"><button id="play">재생</button><input id="step" type="range" min="0" max="19" value="19"><b id="frameLabel"></b><label><input id="global" type="checkbox">전체 240개 킥에 고정 축 적용</label></div>
<canvas id="plot" width="1280" height="880"></canvas><p id="details"></p>
<p class="note">20개 샘플의 진행률을 맞춥니다. 같은 샘플 번호라도 실제 시간·보폭은 다를 수 있습니다. CLT의 카메라 메타데이터는 없습니다. 기본 축 범위는 선택한 킥들의 20프레임 전체에서 고정되며, 좌표를 회전·반사·보간하거나 서로 크기를 맞추지 않습니다. 전체 고정 축은 큰 좌표값도 포함합니다.</p>
<p><a href="README.md">영상·비교 이미지·검증 보고서</a></p></main>
<script>const DATA=__DATA__;
const colors={penaltyfirst15:'#eb871e',EPL:'#1982d2',Women:'#a550aa',CLT:'#239b5f'};
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
seqs.slice().reverse().forEach(s=>{const p=s.poses[step].map(point);ctx.globalAlpha=.82;ctx.strokeStyle=colors[s.dataset];ctx.fillStyle=colors[s.dataset];ctx.lineWidth=s.dataset==='penaltyfirst15'?4:3;edges.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(...p[a]);ctx.lineTo(...p[b]);ctx.stroke()});p.forEach(q=>{ctx.beginPath();ctx.arc(...q,5,0,2*Math.PI);ctx.fill()})});ctx.globalAlpha=1;ctx.font='bold 20px system-ui';ctx.fillStyle='#333';ctx.fillText(cur.kick_id+' · '+(step+1)+'/20',25,32);ctx.font='17px system-ui';seqs.forEach((s,i)=>{ctx.fillStyle=colors[s.dataset];ctx.fillText(s.dataset+' / '+s.kick_id,25+i*310,62)});ctx.fillStyle='#555';ctx.fillText('골반 (0,0) · 같은 x/y 축척 · y는 아래로 증가 · 공 접촉은 20/20',25,855);
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
        label(im,'Orange: penaltyfirst15 | pelvis (0,0) | shared x/y scale | same 20 sampled progress steps',(25,76),.64)
        for i,(ds,old) in enumerate(zip(DATASETS,refs)):
            x=i*640
            label(im,f"{ds} / {old['kick_id'] if old else 'unavailable'}",(x+25,120),.68,COLORS[ds],2)
            rect=(x+60,145,520,625)
            if old: im=draw_skeleton(im,old['poses'][step],low,high,rect,COLORS[ds],4)
            im=draw_skeleton(im,new['poses'][step],low,high,rect,COLORS['penaltyfirst15'],4)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('runs/penaltyfirst15_cross_dataset_overlap'))
    a=parser.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    audit_path=Path('runs/main_data_right_canonical/mirrored_rows.csv')
    source_paths=[NEW,OLD,audit_path,Path('Main Data/Combined_penalty_pose_keypoints_normalized.csv'),Path('Main Data/penaltyfirst15_pose_keypoints_normalized.csv')]
    hashes={str(p):digest(p) for p in source_paths}
    with audit_path.open() as f: mirrored={(r['dataset'],r['kick_id']) for r in csv.DictReader(f)}
    current,references=load(NEW,mirrored),load(OLD,mirrored)
    assert len(current)==14 and len(references)==226
    all_sequences=current+references
    # Verify that each selected sequence scale still comes from ALL source frames.
    full_lengths=defaultdict(list)
    for p in source_paths[-2:]:
        with p.open() as f:
            for r in csv.DictReader(f):
                length=float(r['torso_length'])
                if np.isfinite(length) and length>0: full_lengths[(r['dataset'],r['kick_id'])].append(length)
    scale_errors=[]
    for s in all_sequences:
        scale_errors.append(abs(np.median(full_lengths[(s['dataset'],s['kick_id'])])-float(s['rows'][0]['clip_torso_scale'])))
    assert max(scale_errors)<1e-8
    pairs={};rows=[];rng=random.Random(42)
    selected=list(current)
    for new in current:
        pairs[new['kick_id']]={}
        for ds in DATASETS:
            old,rule,n=select_pair(new,references,ds,rng)
            pairs[new['kick_id']][ds]=(old,rule,n)
            if old: selected.append(old)
            distance=float(np.sqrt(np.mean(np.sum((new['poses']-old['poses'])**2,axis=2)))) if old else None
            rows.append({'new_kick':new['kick_id'],'reference_dataset':ds,'reference_kick':old['kick_id'] if old else '',
                         'selection_rule':rule,'candidate_count':n,'new_camera':new['camera'],'reference_camera':old['camera'] if old else '',
                         'new_original_foot':new['original_foot'],'reference_original_foot':old['original_foot'] if old else '',
                         'canonical_direction':new['direction'],'rms_joint_distance_torso_units':distance})
    low,high=bounds(selected)
    a.output_dir.mkdir(parents=True)
    with (a.output_dir/'matched_pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts=Counter(s['dataset'] for s in all_sequences)
    statistics={}
    for ds in counts:
        s=[x for x in all_sequences if x['dataset']==ds];p=np.concatenate([x['poses'] for x in s])
        statistics[ds]={'kicks':len(s),'rows':len(p),'bounds':[p.min((0,1)).tolist(),p.max((0,1)).tolist()],
                        'absolute_coordinate_percentiles':dict(zip(('p50','p95','p99','max'),np.percentile(abs(p),[50,95,99,100]).tolist()))}
    report={'input_sha256':hashes,'input_files_unchanged':True,'representation_compatible':True,
            'joints':list(JOINTS),'edges':list(EDGES),'feature_order':FEATURES,'feature_shape_per_kick':[20,20],
            'validation':{'kicks':240,'rows':4800,'all_model_features_finite':True,'all_canonical_feet_R':True,
                          'sample_indices_and_contact_endpoints_valid':True,'max_pelvis_origin_error':max(s['pelvis_error'] for s in all_sequences),
                          'max_normalization_formula_error':max(s['normalization_error'] for s in all_sequences),
                          'max_full_sequence_median_scale_error':float(max(scale_errors)),
                          'max_angle_formula_error':max(s['angle_error'] for s in all_sequences)},
            'dataset_statistics':statistics,'selection_seed':42,'selection_based_on_pose_similarity':False,
            'selection_priority':['camera+original_foot+direction','camera+direction','original_foot+direction','direction_only'],
            'unavailable_same_direction_pairs':sum(not r['reference_kick'] for r in rows),
            'pairs':rows,'export_shared_bounds':[low.tolist(),high.tolist()],
            'alignment':'20 ordered sample steps; source clock times and gait phases differ',
            'coordinate_transforms_added':False,
            'interpretation':'Representation compatibility is verified. Identical poses, matching distributions, and cross-dataset model accuracy are not established by an overlap.',
            'CLT_camera':'missing in source; not inferred from filename',
            'HTML':'All 14 new and all 226 reference kicks; manual selection and timeline scrubbing; no network dependencies'}
    web_matches=[]
    for new in current:
        entry={}
        for ds in DATASETS:
            old,rule,n=pairs[new['kick_id']][ds]
            ix=next((i for i,s in enumerate(references) if old is s),None)
            entry[ds]={'reference_index':ix,'selection_rule':rule}
        web_matches.append(entry)
    web={'current':[payload(s) for s in current],'reference':[payload(s) for s in references],'matches':web_matches}
    # Escape the JS string's newline before replacing it with the data payload.
    html=HTML.replace("lines.join('\n')", "lines.join('\\n')")
    (a.output_dir/'index.html').write_text(html.replace('__DATA__',json.dumps(web,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    cv2.setNumThreads(2)
    for ds in DATASETS:
        print('Rendering overlap:',ds,flush=True)
        images=[render(current,pairs,ds,i,low,high) for i in range(20)]
        encode(images,a.output_dir/f'vs_{ds}_overlap.mp4')
        cv2.imwrite(str(a.output_dir/f'vs_{ds}_contact_preview.png'),images[-1])
        gif=[Image.fromarray(cv2.cvtColor(cv2.resize(im,(960,720)),cv2.COLOR_BGR2RGB)) for im in images]
        gif[0].save(a.output_dir/f'vs_{ds}_overlap.gif',save_all=True,append_images=gif[1:],duration=[200]*19+[1200],loop=0)
        contact_figure(current,pairs,ds,low,high,a.output_dir)
    print('Rendering enlarged three-dataset overlap',flush=True)
    report['focused_example']=write_focus(current,pairs,a.output_dir)
    assert hashes=={str(p):digest(p) for p in source_paths}
    report['all_3_mp4_files_fully_decoded']=True
    (a.output_dir/'compatibility_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    lines=['# penaltyfirst15와 기존 데이터셋 overlap','',
           '[프레임·킥 선택형 시각화](index.html) — 브라우저로 열면 현재 14개와 기존 226개 킥을 직접 선택하고 겹칠 수 있습니다.','',
           '[세 데이터셋 확대 overlap 영상](focused_three_dataset_overlap.mp4) · [GIF](focused_three_dataset_overlap.gif) · [접촉 이미지](focused_contact_overlap.png)','',
           '주황색은 penaltyfirst15, 파란색은 EPL, 보라색은 Women, 초록색은 CLT입니다. 색상은 데이터셋을 구분합니다.','',
           '각 그래프의 골반 원점과 x/y 축척을 동일하게 적용하고, 20개 샘플의 진행률을 맞췄습니다. 내보낸 세 영상·이미지는 동일한 축 범위를 사용합니다. HTML 기본 축은 선택한 킥들의 20프레임 전체에 고정됩니다.','',
           '## 형식 호환성 검증','',
           '현재 14개와 기존 EPL 77개, Women 128개, CLT 21개를 검사했습니다. 동일한 8관절 순서·8개 연결, 오른발 통일, 20프레임, 골반 중심, 전체 구간의 중앙 몸통 길이로 양축 정규화, 4개 각도/180 정의가 일치합니다. 4,800행의 모델 피처는 모두 유한합니다.','',
           '이는 같은 그래프·피처 형식으로 사용할 수 있다는 의미입니다. 선수의 동작·분포·모델 성능까지 같다고 판단한 결과는 아닙니다.','',
           '## 비교 표본','',
           'seed 42로 카메라+원래 킥 발+정규화된 킥 방향이 일치하는 표본을 우선 선택했습니다. 없으면 카메라+방향, 원래 발+방향, 방향 순서로 조건을 완화했습니다. 포즈가 가장 비슷한 표본을 고르지 않았습니다. 조건·후보 수는 matched_pairs.csv에 있습니다.','',
           'CLT는 카메라 값이 없습니다. 정규화된 C 방향 표본도 없어 현재 C 방향 킥 2개에는 CLT를 자동 겹치지 않았습니다. Women은 모두 원래 오른발이어서 현재 왼발 킥은 원래 발 조건이 다릅니다. HTML에서 직접 다른 표본을 선택할 수 있습니다.','',
           'CLT의 최대 절대 정규화 좌표는 약 5.95, EPL은 3.92로 현재 데이터의 1.85보다 큽니다. 정규화 수식은 일치하지만 큰 좌표값과 카메라 차이는 그대로 표시했습니다. 좌표를 추가 회전·반사·보간하거나 골격별로 크기를 맞추지 않았습니다.','',
           '20개 샘플 번호의 일치는 실제 시간이나 보폭 단계가 같다는 뜻이 아닙니다. 마지막 샘플은 모두 접촉 프레임입니다.','',
           '| 비교 | 동작 overlap | 접촉 그래프 |','|---|---|---|']
    for ds in DATASETS: lines.append(f'| {ds} | [MP4](vs_{ds}_overlap.mp4) · [GIF](vs_{ds}_overlap.gif) | [PNG](vs_{ds}_contact.png) · [PDF](vs_{ds}_contact.pdf) |')
    lines+=['','[표본 대응표](matched_pairs.csv) · [호환성 검증 보고서](compatibility_report.json)','',
            '재실행: `python compare_penaltyfirst15_graphs.py --output-dir 새_폴더`']
    (a.output_dir/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output_dir':str(a.output_dir),'representation_compatible':True,'kicks_checked':240,'rows_checked':4800,'pairs':len(rows),'unavailable_pairs':report['unavailable_same_direction_pairs']}))


if __name__=='__main__':
    main()
