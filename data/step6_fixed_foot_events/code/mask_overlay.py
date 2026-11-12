"""Display original Step4 RLE regions, without constructing masks from keypoints."""
import cv2
import numpy as np

REGION_COLORS = [(190, 55, 215), (20, 185, 245), (225, 210, 35)]  # BGR
REGION_NAMES = ["heel", "mid", "toe"]


def foreground_bounds(runs, width, height):
    if not len(runs):
        return None
    ends = np.cumsum(runs, dtype=np.int64)
    if int(ends[-1]) != width * height:
        raise ValueError("Step4 RLE canvas differs from oriented source image")
    starts = np.r_[0, ends[:-1]]
    begin, stop = starts[1::2], ends[1::2]
    good = stop > begin
    begin, stop = begin[good], stop[good]
    if not len(begin):
        return None
    crosses_row = begin // width != (stop - 1) // width
    x0 = 0 if crosses_row.any() else int((begin % width).min())
    x1 = width if crosses_row.any() else int(((stop - 1) % width).max() + 1)
    return x0, int((begin // width).min()), x1, int(((stop - 1) // width).max() + 1)


def decode_crop(runs, width, height, box):
    x0, y0, x1, y1 = box
    if not len(runs):
        return np.zeros((y1 - y0, x1 - x0), bool)
    ends = np.cumsum(runs, dtype=np.int64)
    if int(ends[-1]) != width * height:
        raise ValueError("RLE canvas mismatch")
    positions = np.arange(y0, y1)[:, None] * width + np.arange(x0, x1)[None]
    return np.searchsorted(ends, positions, side="right") % 2 == 1


def prepare_regions(payload, row, slot, width, height):
    region_runs, boxes = [], []
    for kind in (1, 2, 3):
        begin, end = payload["rle_ptr"][row, slot, kind]
        runs = payload["rle_counts"][int(begin):int(end)]
        region_runs.append(runs)
        bounds = foreground_bounds(runs, width, height)
        if bounds is not None:
            boxes.append(bounds)
    if not boxes:
        return None
    box = (max(0, min(b[0] for b in boxes) - 3), max(0, min(b[1] for b in boxes) - 3),
           min(width, max(b[2] for b in boxes) + 3), min(height, max(b[3] for b in boxes) + 3))
    masks = [decode_crop(runs, width, height, box) for runs in region_runs]
    return box, masks


def apply_regions(image, prepared):
    if prepared is None:
        return image.copy(), None
    box, masks = prepared
    x0, y0, x1, y1 = box
    result = image.copy()
    roi = result[y0:y1, x0:x1]
    for mask, color in zip(masks, REGION_COLORS):
        roi[mask] = np.round(.55 * roi[mask] + .45 * np.array(color)).astype(np.uint8)
    union = np.logical_or.reduce(masks).astype(np.uint8)
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi, contours, -1, (245, 245, 245), 1, cv2.LINE_AA)
    pad = max(15, round(.25 * max(x1 - x0, y1 - y0)))
    return result, result[max(0, y0-pad):min(image.shape[0], y1+pad), max(0, x0-pad):min(image.shape[1], x1+pad)].copy()
