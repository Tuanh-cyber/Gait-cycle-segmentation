"""GMA và nhất quán forward/backward."""
import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import s5_common as C

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")


def build(device):
    sys.path.insert(0, str(C.GMA_ROOT / "core"))
    from network import RAFTGMA
    args = argparse.Namespace(model=str(C.CHECKPOINT), model_name=C.MODEL, num_heads=1, position_only=False, position_and_content=False,
                              mixed_precision=False, alternate_corr=False, small=False, dropout=0)
    m = RAFTGMA(args)
    sd = torch.load(C.CHECKPOINT, map_location="cpu", weights_only=True)
    m.load_state_dict({k.replace("module.", "", 1): w for k, w in sd.items()}, strict=True)
    return m.to(device).eval()


def to_tensor(img, device):
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(device)


@torch.inference_mode()
def flow_pair(model, a, b, device):
    """a, b: BGR cùng kích thước. Trả về flow forward, backward (H×W×2, float32)."""
    h, w = a.shape[:2]
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = F.pad(to_tensor(a, device)[None], (0, pad_w, 0, pad_h), mode="replicate")
    y = F.pad(to_tensor(b, device)[None], (0, pad_w, 0, pad_h), mode="replicate")
    _, f = model(torch.cat([x, y]), torch.cat([y, x]), iters=C.ITERS, test_mode=True)
    f = f[:, :, :h, :w].permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    return f[0], f[1]


def fb_valid(fw, bw):
    """Pixel hợp lệ: |f(x) + b(x + f(x))| <= FB_ABS + FB_REL × |f(x)|, và x + f(x) còn trong ảnh."""
    h, w = fw.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xx + fw[..., 0], yy + fw[..., 1]
    b = cv2.remap(bw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.linalg.norm(fw + b, axis=2)
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return (err <= C.FB_ABS + C.FB_REL * np.linalg.norm(fw, axis=2)) & inside
