#!/usr/bin/env python3
"""Draw the archived initial 20-frame / 20-feature model, without training.

Tensor shapes are measured by a CPU forward pass through the saved code.
This diagram deliberately describes the initial 32/32/64-channel model,
not the later smaller 16/16/32-channel cross-validation models.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/main_data_8j20_angles_lr_stgcn_seed42"
OUT = ROOT / "runs/visualizations"


def import_snapshot(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_archived_model():
    """Load both archived files and verify the drawing's dimensions."""
    saved_model_module = sys.modules.get("model")
    try:
        sys.modules["model"] = import_snapshot(RUN / "model_blocks_code.py", "archived_blocks")
        training = import_snapshot(RUN / "training_code.py", "archived_training")
    finally:
        if saved_model_module is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = saved_model_module
    info = json.loads((RUN / "run_info.json").read_text())
    model = training.PoseAngleSTGCN(info["config"]["dropout"]).eval()
    assert str(model).strip() == (RUN / "model_architecture.txt").read_text().strip()
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == info["model"]["parameters"] == 44778
    shapes = {}
    hooks = []
    for name, module in [("coordinate_norm", model.coordinate_norm),
                         *[(f"block_{i+1}", b) for i, b in enumerate(model.blocks)],
                         ("angle_norm", model.angle_branch[0]),
                         ("angle_conv_1", model.angle_branch[1]),
                         ("angle_conv_2", model.angle_branch[4]),
                         ("angle_pool", model.angle_branch[6]),
                         ("classifier", model.head[1])]:
        def record(module, inputs, output, key=name):
            shapes[key] = {"input": list(inputs[0].shape), "output": list(output.shape)}
        hooks.append(module.register_forward_hook(record))
    torch.set_num_threads(2)
    with torch.no_grad():
        logits = model(torch.zeros(2, 20, 20))
    for hook in hooks:
        hook.remove()
    expected = {"coordinate_norm": [2, 16, 20], "block_1": [2, 32, 20, 8],
                "block_2": [2, 32, 20, 8], "block_3": [2, 64, 10, 8],
                "angle_norm": [2, 4, 20], "angle_conv_1": [2, 16, 20],
                "angle_conv_2": [2, 16, 20], "angle_pool": [2, 16, 1],
                "classifier": [2, 2]}
    assert {k: v["output"] for k, v in shapes.items()} == expected
    assert shapes["classifier"]["input"] == [2, 80] and list(logits.shape) == [2, 2]
    return {"status": "PASS", "source_run": str(RUN), "parameters": parameters,
            "feature_columns": info["feature_columns"], "batch_size_for_shape_check": 2,
            "forward_shapes": shapes, "label_mapping": info["label_mapping"],
            "note": "Shape-check only, no training or evaluation; Softmax is inference postprocessing."}


def draw_diagram():
    fig, ax = plt.subplots(figsize=(16, 15))
    fig.patch.set_facecolor("#f7fafc")
    ax.set(xlim=(0, 18), ylim=(0, 17))
    ax.axis("off")
    blue, amber, navy = "#dceeff", "#fff0d5", "#15354b"

    def label(x, y, text, size=12, weight="normal", color=navy):
        ax.text(x, y, text, ha="center", va="center", fontsize=size,
                fontweight=weight, color=color, linespacing=1.4)

    def box(x, y, w, h, text, fill, size=12):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.06,rounding_size=0.13",
                     linewidth=1.2, edgecolor="#8aa0b2", facecolor=fill, zorder=2))
        label(x, y, text, size)

    def arrow(start, end, color="#506b80"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                     mutation_scale=14, linewidth=1.6, color=color,
                     connectionstyle="arc3", zorder=1))

    label(9, 16.5, "Initial ST-GCN: 20 frames × 20 features", 23, "bold")
    label(9, 15.93, "Archived model: PoseAngleSTGCN  |  44,778 trainable parameters  |  Kick direction: L / R", 11)
    box(9, 14.92, 11, 1.1,
        "Input  [B, 20, 20]\n16 normalized coordinates + 4 normalized joint angles per frame", "#e8edf3", 13)
    label(4.65, 13.78, "COORDINATE / GRAPH BRANCH", 12, "bold", "#22679b")
    label(13.35, 13.78, "ANGLE / TEMPORAL BRANCH", 12, "bold", "#96610a")
    arrow((7, 14.31), (4.65, 13.39)); arrow((11, 14.31), (13.35, 13.39))
    left = [
        (12.85, "16 coordinates → BatchNorm1d(16) → reshape\n[B, 2, 20, 8] = [batch, channels, time, joints]"),
        (11.60, "ST-GCN block 1: 2 → 32 channels\nTemporal kernel 5, stride 1   →   [B, 32, 20, 8]"),
        (10.35, "ST-GCN block 2: 32 → 32 channels\nTemporal kernel 5, stride 1   →   [B, 32, 20, 8]"),
        (9.10, "ST-GCN block 3: 32 → 64 channels\nTemporal kernel 5, stride 2   →   [B, 64, 10, 8]"),
        (7.85, "Global average pooling over time AND joints\n[B, 64, 10, 8] → [B, 64]"),
    ]
    right = [
        (12.85, "4 angles → transpose → BatchNorm1d(4)\n[B, 4, 20] = [batch, channels, time]"),
        (11.60, "Conv1d 4 → 16  |  kernel 5, padding 2\nReLU → Dropout(0.3)   →   [B, 16, 20]"),
        (10.35, "Conv1d 16 → 16  |  kernel 5, padding 2\nReLU   →   [B, 16, 20]"),
        (9.10, "AdaptiveAvgPool1d(1) over time\n[B, 16, 20] → [B, 16, 1]"),
        (7.85, "Squeeze final dimension\n[B, 16, 1] → [B, 16]"),
    ]
    for x, stages, color in [(4.65, left, blue), (13.35, right, amber)]:
        for i, (y, text) in enumerate(stages):
            box(x, y, 7.8, .91, text, color, 11.4)
            if i:
                arrow((x, stages[i-1][0]-.52), (x, y+.52))
    arrow((4.65, 7.32), (7.4, 7.04)); arrow((13.35, 7.32), (10.6, 7.04))
    box(9, 6.65, 8.5, .76, "Concatenate: 64 coordinate features + 16 angle features  →  [B, 80]", "#e7e1f7", 12)
    arrow((9, 6.19), (9, 5.98))
    box(9, 5.5, 8.5, .8, "Dropout(0.3) → Linear(80 → 2)\nL / R logits  [B, 2]", "#e7e1f7", 12)
    arrow((9, 5.03), (9, 4.86))
    box(9, 4.43, 8.5, .68, "Inference only: Softmax → P(L), P(R) → argmax", "#dcefe5", 12)

    # Two explanatory insets: an exact block schematic and the actual graph.
    box(5.6, 2.12, 9.65, 3.0, "", "#ffffff")
    label(5.6, 3.32, "Inside each ST-GCN block", 13, "bold")
    box(2.55, 2.44, 2.85, .71, "1×1 Conv2d\nCin → 3 × Cout", blue, 10.3)
    box(5.82, 2.44, 2.9, .71, "3 graph partitions\naggregate → Cout", blue, 10.3)
    box(9, 2.44, 2.55, .71, "Temporal Conv\nBN → ReLU → k=5\n→ BN → Dropout", blue, 9.1)
    arrow((4.03, 2.44), (4.30, 2.44)); arrow((7.34, 2.44), (7.67, 2.44))
    label(5.6, 1.60, "+ residual connection → ReLU", 11.5, "bold")
    label(5.6, 1.04, "Partitions: equal / closer / farther from hips; incoming-degree normalized.\nResidual: identity in block 2; 1×1 Conv + BN in blocks 1 and 3.", 9.3)

    box(14.17, 2.12, 6.08, 3.0, "", "#ffffff")
    label(14.17, 3.32, "8-joint skeleton graph", 13, "bold")
    points = {5: (13.1, 2.83), 6: (15.25, 2.83), 11: (13.4, 2.13),
              12: (14.95, 2.13), 13: (13.3, 1.5), 14: (15.05, 1.5),
              15: (13.2, .9), 16: (15.15, .9)}
    for a, b in [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
        ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color="#648aaf", lw=2.4, zorder=3)
    for j, (x, y) in points.items():
        ax.scatter([x], [y], s=65, c="#276da5", zorder=4)
        ax.text(x+(-.27 if j % 2 else .27), y, str(j), ha="right" if j % 2 else "left",
                va="center", fontsize=10, color=navy)
    label(12.05, 2.78, "Shoulder", 8.5); label(12.05, 2.12, "Hip", 8.5)
    label(12.05, 1.50, "Knee", 8.5); label(12.05, .90, "Ankle", 8.5)
    label(9, .25, "B = batch size  •  angles = left/right knee + left/right hip  •  metadata and goal are NOT model inputs", 10)
    fig.subplots_adjust(left=.015, right=.985, bottom=.01, top=.99)
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "initial_stgcn_20frames_20features_architecture"
    fig.savefig(base.with_suffix(".svg"), facecolor=fig.get_facecolor())
    fig.savefig(base.with_suffix(".png"), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base


if __name__ == "__main__":
    verification = verify_archived_model()
    base = draw_diagram()
    base.with_suffix(".json").write_text(json.dumps(verification, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"png": str(base.with_suffix(".png")), "svg": str(base.with_suffix(".svg")),
                      "validation": verification}, indent=2))
