# 전체 프레임 spatial normalization

`spatial_normalize_pose.py`는 보간과 오른발 기준 좌우 반전이 완료된 CSV를
입력받습니다. 원본 행·행 순서·열 순서·메타데이터·좌표 문자열을 유지하고,
12개 관절(`kp_5`–`kp_16`)의 `_xn`, `_yn` 24열과 진단용 6열만 추가합니다.

각 행에서 골반 중심 H와 어깨 중심 S를 계산하고, 유클리드 거리
`d = hypot(S_x-H_x, S_y-H_y)`를 구합니다. 각 kick의 **전체 행** 중 양수이며
유한한 d의 median을 고정 scale D로 사용합니다. 새 좌표는 `(p-H)/D`입니다.
회전, bbox 정규화, 재보간, 좌우 반전, 라벨 변경, 프레임 선택은 하지 않습니다.

## 실행

```bash
python spatial_normalize_pose.py
```

기본 실행은 `Main Data`의 CLT, EPL, Women, Combined CSV를 처리하고 같은 폴더에
각각 `_normalized.csv`와 `_normalized_normalization_report.json`을 저장합니다.
기존 출력이 있으면 중단하며, 의도적으로 재생성할 때만 `--overwrite`를 사용합니다.

특정 파일만 처리하려면:

```bash
python spatial_normalize_pose.py "Main Data/Combined_penalty_pose_keypoints.csv"
```

함수를 직접 호출할 수도 있습니다.

```python
from spatial_normalize_pose import normalize_pose_csv

report = normalize_pose_csv("input.csv", "output.csv")
```

단일 데이터셋 파일은 `kick_id`별, `dataset` 열이 있는 통합 파일은
`(dataset, kick_id)`별로 scale을 계산합니다. 요청한 ID 단위를 유지하므로
트랙이나 카메라를 기준으로 별도의 scale을 나누지 않습니다. 같은 ID에 여러
트랙/카메라가 있으면 해당 ID의 모든 행이 같은 D를 사용합니다.

## 실패 및 검증

유효 torso length가 없는 kick은 warning과 JSON 보고서의 `failed_kicks`에
기록합니다. 그 kick의 행은 유지하고 새로운 scale/정규화 좌표만 빈칸
(CSV의 NaN)으로 저장합니다. bbox를 대체 scale로 사용하지 않습니다.
개별 관절이 누락됐을 때도 해당 정규화 좌표는 NaN으로 남깁니다.
`xc/yc` 값 0은 유효한 상대좌표이므로 자동으로 결측 취급하지 않습니다.

각 파일의 보고서와 콘솔에 행/ID 수, 실패 목록, clip scale의 min/median/max,
골반 중심 최대 오차, 새 좌표의 NaN 수, 고정 scale 여부 및 무작위 kick 예시를
기록합니다. scale 통계는 행 수 가중 없이 kick별 하나의 D에 대해 계산합니다.
예시의 첫 프레임은 정렬된 frame 번호가 아니라 원본에 처음 등장한 행입니다.
기존 열 전체를 저장된 CSV와 행 단위로 재대조하고 원본 파일 SHA-256도 확인합니다.

```bash
python -m unittest -v test_spatial_normalize_pose.py
```

테스트는 20행 이후의 데이터까지 median에 반영하는지, 순서와 메타데이터를
보존하는지, 기울기를 유지하는지, 데이터셋 간 ID 충돌을 분리하는지,
유효 scale이 없는 kick을 유지하고 기록하는지 확인합니다.
