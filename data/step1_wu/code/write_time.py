"""Ghi trục thời gian thật vào frames CSV + tóm tắt vào metadata.
Cột mới trong step0_audit/frames/<vid>_frames.csv:
  speed_part_id  : chỉ số phần tốc độ (-1 = freeze biên tập / shot bị loại / ngoài mọi phần)
  speed_k        : hệ số (t_real = t_video × k)
  k_confidence   : cao / trung bình / thấp
  continuity_id  : khối liên tục về thời gian; cycle KHÔNG được vắt qua 2 khối (cắt cảnh / freeze)
  t_real_s       : thời gian thật (s), cộng dồn dt×k; freeze bị thu gọn (không tính thời gian)"""
import pandas as pd, numpy as np, openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
P = pd.read_csv("step1_wu/speed_parts.csv")
meta = pd.read_csv("step0_audit/step0_metadata.csv")
summ = []
for v in meta.video_id:
    f = f"step0_audit/frames/{v}_frames.csv"; fr = pd.read_csv(f)
    pv = P[P.video_id == v].sort_values("t0").reset_index(drop=True)
    pid = np.full(len(fr), -1); kk = np.full(len(fr), np.nan); conf = np.array([""] * len(fr), dtype=object)
    for i, r in pv.iterrows():
        m = (fr.pts_s >= r.t0 - 1e-6) & (fr.pts_s < r.t1 - 1e-6) & (~fr.is_edit_freeze)
        pid[m] = i; kk[m] = r.k; conf[m] = r.k_confidence
    seg_of_part = pv.seg.values
    cont = np.where(pid >= 0, seg_of_part[np.maximum(pid, 0)], -1)
    t_real = np.full(len(fr), np.nan); acc = 0.0; prev_t = None; prev_c = None
    for i in range(len(fr)):
        if pid[i] < 0: continue
        if prev_t is not None and cont[i] == prev_c: acc += (fr.pts_s.iloc[i] - prev_t) * kk[i]
        t_real[i] = acc; prev_t = fr.pts_s.iloc[i]; prev_c = cont[i]
    fr["speed_part_id"] = pid; fr["speed_k"] = kk; fr["k_confidence"] = conf
    fr["continuity_id"] = cont; fr["t_real_s"] = np.round(t_real, 5)
    fr.to_csv(f, index=False)
    lab = "; ".join(f"{r.t0:.2f}–{r.t1:.2f}s: k={r.k_label} ({r.k_confidence})" for r in pv.itertuples())
    good = pv[pv.k_confidence.isin(["cao", "trung bình"])]
    cad = np.average(good.cadence_real_spm, weights=good.dur_video_s) if len(good) else np.nan
    rank = {"cao": 2, "trung bình": 1, "thấp": 0}
    summ.append(dict(video_id=v, n_speed_parts=len(pv), speed_parts=lab,
                     has_slowmo=bool((pv.k < 0.99).any()),
                     realtime_fraction=round(pv[pv.k >= 0.99].dur_video_s.sum() / max(pv.dur_video_s.sum(), 1e-9), 2),
                     k_min_confidence=min(pv.k_confidence, key=lambda c: rank[c]) if len(pv) else "",
                     cadence_real_spm=round(cad, 1) if np.isfinite(cad) else np.nan,
                     abs_time_ok=bool(len(pv) and (pv.k_confidence != "thấp").all()),
                     speed_flags="; ".join(sorted({x for fl in pv["flags"].fillna("") for x in fl.split("; ") if x}))))
S = pd.DataFrame(summ)
for c in S.columns[1:]:
    if c in meta.columns: meta = meta.drop(columns=c)
meta = meta.merge(S, on="video_id"); meta.to_csv("step0_audit/step0_metadata.csv", index=False, encoding="utf-8-sig")
wb = openpyxl.load_workbook("step0_audit/step0_metadata.xlsx"); ws = wb["metadata"]
hdr = [c.value for c in ws[1]]; vcol = hdr.index("video_id") + 1; L = S.set_index("video_id")
red = PatternFill("solid", fgColor="F8D7DA"); green = PatternFill("solid", fgColor="D4EDDA")
for c in S.columns[1:]:
    if c in hdr: col = hdr.index(c) + 1
    else:
        col = ws.max_column + 1; h = ws.cell(1, col, c); hdr.append(c)
        h.font = Font(bold=True, color="FFFFFF"); h.fill = PatternFill("solid", fgColor="1F4E78"); h.alignment = Alignment(wrap_text=True, vertical="top")
    for r in range(2, ws.max_row + 1):
        val = L.loc[ws.cell(r, vcol).value, c]
        if isinstance(val, (np.bool_,)): val = bool(val)
        if isinstance(val, float) and np.isnan(val): val = None
        cell = ws.cell(r, col, val)
        if c == "abs_time_ok": cell.fill = green if val else red
    ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 70 if c in ("speed_parts", "speed_flags") else 14
rd = wb["README"]
for a, b in [
 ("speed_parts", "Các phần tốc độ (thời gian video gốc): k = hệ số thời gian thật/thời gian video. Nguồn: step1_wu/speed_parts.csv"),
 ("cách chốt k", "Wu et al. (2026) chỉ dùng để phân loại & chọn giữa ứng viên; giá trị Wu liên tục nhiễu ±40% trên dataset này. k lấy từ: (a) tỉ lệ cadence phần chậm/phần nhanh trong cùng clip; (b) ứng viên {1,1/2,1/3,1/4,1/5,1/6,1/8} cho cadence thực 140–215 spm, gần Wu nhất"),
 ("k_confidence", "cao = real-time, ≥4 bước, peak rõ; trung bình = có ≥4 bước hoặc tỉ lệ cadence đo được; thấp = phần slow-mo <4 bước / mâu thuẫn Wu / nhiều ứng viên"),
 ("abs_time_ok", "True nếu mọi phần có k_confidence ≠ thấp. Đặc trưng thời gian TUYỆT ĐỐI (cadence, contact time, flight time) chỉ dùng video True. Đặc trưng chuẩn hóa theo pha chu kỳ (%cycle) dùng được cho mọi video, miễn cycle không vắt qua 2 speed_part"),
 ("frames CSV (mới)", "speed_part_id, speed_k, k_confidence, continuity_id, t_real_s — xem docstring step1_wu/code/write_time.py"),
 ("cadence_real_spm", "Cadence thực trung bình (trọng số thời lượng) trên các phần có độ tin cậy ≥ trung bình")]:
    rd.append([a, b])
wb.save("step0_audit/step0_metadata.xlsx")
print(S[["video_id","n_speed_parts","has_slowmo","realtime_fraction","k_min_confidence","cadence_real_spm","abs_time_ok"]].to_string())
print(S.abs_time_ok.value_counts()); print(S.has_slowmo.value_counts())
