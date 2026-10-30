"""Run the final pre3 checkpoint on already normalized NumPy inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import PenaltySTGCN, LABELS, FEATURES

ROOT = Path(__file__).resolve().parent


def load_model(checkpoint=ROOT / 'weights/final_pre3.pt'):
    # The published, locally trained checkpoint contains NumPy metadata.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['features'] != list(FEATURES):
        raise ValueError('Checkpoint feature order does not match the final model')
    model = PenaltySTGCN(saved['config']['dropout'])
    model.load_state_dict(saved['model'], strict=True)
    return model.eval()


def predict(inputs, checkpoint=ROOT / 'weights/final_pre3.pt'):
    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or x.shape[1:] != (3, 12) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite normalized coordinates with shape (N, 3, 12) or (3, 12)')
    model = load_model(checkpoint)
    with torch.no_grad():
        probabilities = model(torch.from_numpy(x)).softmax(dim=1).numpy()
    return [{'label': LABELS[int(p.argmax())], 'prob_L': float(p[0]), 'prob_R': float(p[1])}
            for p in probabilities]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Preprocessed (N,3,12) or (3,12) .npy array')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'weights/final_pre3.pt')
    args = parser.parse_args()
    print(json.dumps(predict(np.load(args.input, allow_pickle=False), args.checkpoint), indent=2))
