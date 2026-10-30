# Penalty Kick Prediction Model

YSAL에서 개발한 페널티킥 포즈 전처리와 딥러닝 실험 코드입니다. ST-GCN, CNN,
관절·프레임 수 비교, 접촉 이전 구간 실험 및 학습된 체크포인트를 포함합니다.

## 설치와 검증

Python 3.10 환경에서 기존 단위 테스트 38개를 통과했습니다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
```

`requirements.txt`는 직접 의존성 목록이며 버전을 고정한 환경 잠금 파일은 아닙니다.
영상 처리에는 별도 FFmpeg 실행 파일이 필요할 수 있습니다. GPU 종류와 개수는
설정 파일의 `device` / `devices`를 실행 환경에 맞게 변경하세요.

## 전처리 코드

| 단계 | 주요 파일 |
| --- | --- |
| 포즈 추출 및 검토 | `extract_yolov8_pose.py`, `extract_pose_ball_contact.py`, `reviewed_pose_inference.py` |
| 데이터셋별 처리 | `preprocess_150penalties.py`, `preprocess_penaltyfirst15.py` |
| 데이터 통합·발 기준 정렬 | `merge_main_data.py`, `canonicalize_main_data.py` |
| 공간 정규화 | `spatial_normalize_pose.py` |
| 20프레임 샘플링·8관절 선택 | `extract_uniform_20frames.py`, `extract_8_joint_pose.py` |
| 기존 파이프라인의 정규화·보간 | `normalize_combined_20frames.py`, `interpolate_lower_body.py` |

실험마다 입력 CSV와 전처리 방식이 다릅니다. 위 파일은 모든 실험에 공통으로
순서대로 실행하는 단일 파이프라인이 아닙니다. 상세 조건은
[공간 정규화 문서](SPATIAL_NORMALIZATION.md),
[포즈 추출 문서](YOLOV8_POSE_PIPELINE.md) 및 각 설정 파일을 확인하세요.

## 모델과 학습

- `model.py`, `train.py`: 12관절 / 4채널 ST-GCN 및 학습 코드.
- `train_8j20_angles_lr.py`: 8관절 좌표 ST-GCN + 4개 관절각 시계열 분기. 입력은
  `(batch, 20 frames, 20 features)`, 출력은 L/R입니다.
- `compare_8j20_lr_cv.py`: ST-GCN/CNN 교차검증 비교.
- `experiment_*`: 카메라, 프레임·관절 수, 접촉 이전 구간 등의 비교 실험.

원본 영상, 데이터 CSV, 데이터 분할 파일, 예측 결과와 캐시는 배포하지 않습니다.
학습하려면 직접 준비한 데이터를 해당 설정의 입력 경로에 배치해야 합니다.
특히 아래 8관절 학습은 정규화 좌표 16열과 정규화된 무릎·고관절 각도 4열이
이미 들어 있는 CSV를 요구합니다. 정확한 열 순서는 코드의 `FEATURES`를 확인하세요.

```bash
python train_8j20_angles_lr.py --dry-run
python train_8j20_angles_lr.py --config train_8j20_angles_lr_config.json
```

배포된 체크포인트 경로가 이미 있으므로 다시 학습할 때는 설정을 복사하고
`output_dir` 및 `data_dir`를 새 경로로 지정하세요. 데이터가 없는 상태에서는
`--dry-run`도 입력 검증을 완료할 수 없습니다.

기존 연구 문서는 당시 결과물 경로도 참조합니다. 체크포인트를 제외한 실험
결과물은 이번 업로드 범위에 없으므로 해당 보고서를 다시 생성하려면 입력
데이터를 준비하고 관련 실험을 재실행해야 합니다. `DESIGN.md`는 초기 설계 기록이며
현재 구현 상태는 실제 코드와 개별 실험 문서를 기준으로 확인하세요.

## 학습된 가중치

`runs/` 아래 원래 실험/조건/시드/폴드 경로를 유지한 체크포인트 493개
(총 27,925,116 bytes)를 포함합니다. `checkpoints.json`에 경로·크기·SHA-256을
기록했습니다. 체크포인트와 같은 폴더의 코드 스냅샷·모델 구조·`run_info.json`은
원본에 존재하는 경우 함께 포함했습니다. YOLO 사전학습 가중치는 포함하지 않습니다.
`run_info.json`에 기록된 `/workspace/YSAL/...` 경로는 학습 당시의 출처 기록입니다.

8관절/20프레임 L/R 모델의 CPU 로딩 및 형태 확인 예제:

```python
import torch
from train_8j20_angles_lr import PoseAngleSTGCN

# 이 저장소의 신뢰할 수 있는 체크포인트만 로드하세요.
# 일부 원본 체크포인트에는 NumPy 메타데이터가 있어 weights_only=False를 사용합니다.
checkpoint = torch.load(
    "runs/main_data_8j20_angles_lr_stgcn_seed42/best.pt",
    map_location="cpu",
    weights_only=False,
)
model = PoseAngleSTGCN(dropout=checkpoint["config"]["dropout"])
model.load_state_dict(checkpoint["model"])
model.eval()
with torch.no_grad():
    # 형태 확인용 더미 입력이며 실제 예측용 데이터가 아닙니다.
    probabilities = model(torch.zeros(1, 20, 20)).softmax(dim=1)
print(probabilities.shape)  # torch.Size([1, 2]); L=0, R=1
```

다른 실험 체크포인트는 해당 실험의 모델 클래스와 입력 형태를 사용해야 합니다.

## 결과 해석

[8관절 L/R 기준 실험](8J20_LR_TRAINING.md)의 테스트 정확도는 17/32(53.125%)로,
동일 테스트에서 항상 L을 예측하는 기준선과 같습니다. 선수·경기 단위의 일반화는
입증되지 않았습니다. 전체 구간에서 계산한 정규화나 접촉 프레임을 포함하는
실험과 엄격한 접촉 이전 실험을 구분해서 해석해야 합니다.

추가 조건은 [교차검증](8J20_LR_CV_COMPARISON.md),
[카메라](CAMERA_C_EXPERIMENTS.md), [관절·프레임](FRAMES_JOINTS_EXPERIMENTS.md),
[접촉 이전 실험](PRECONTACT_EXPERIMENTS.md) 문서에 기록되어 있습니다.
