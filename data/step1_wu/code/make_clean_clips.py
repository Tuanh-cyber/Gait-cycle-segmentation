"""Cắt clip sạch cho Wu et al. (Seeing Fast and Slow):
 - bỏ các đoạn freeze biên tập (is_edit_freeze)
 - tách theo shot (shot_id)
 - chuyển VFR -> CFR 30 fps theo đúng timestamp thật (-vf fps=30)
 - xuất .mp4 H.264; clip ngắn hơn CLIP_MIN bị đánh dấu too_short (Wu sẽ lỗi chia 0)
Chạy từ thư mục data/:  python step1_wu/code/make_clean_clips.py
"""
import pandas as pd, numpy as np, subprocess, json, os, sys
CLIP_MIN = 2.3          # s; phải > clip_duration = num_frames/7.5 của checkpoint (≈2.13 s nếu num_frames=16)
AUD, OUT = "step0_audit", "step1_wu/clips"
os.makedirs(OUT, exist_ok=True)

def start_time(p):
    s = json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=start_time","-of","json",p],capture_output=True,text=True).stdout)["streams"][0]
    return float(s.get("start_time",0) or 0)

meta = pd.read_csv(f"{AUD}/step0_metadata.csv")
excl = {(r.video_id, int(s.replace("shot",""))) for r in meta.itertuples() if isinstance(r.excluded_shots,str) and r.excluded_shots for s in r.excluded_shots.split(",")}
rows = []
for m in meta.itertuples():
    src = f"{'Correct' if m.label=='CORRECT' else 'False'}/{m.video}"
    fr = pd.read_csv(f"{AUD}/frames/{m.video_id}_frames.csv")
    st = start_time(src)
    keep = ~fr.is_edit_freeze
    seg = ((keep != keep.shift()) | (fr.shot_id != fr.shot_id.shift())).cumsum()
    fr["seg"] = seg
    dt_med = fr.pts_s.diff().median()
    for j, (k, g) in enumerate(fr[keep].groupby("seg")):
        shot = int(g.shot_id.iloc[0])
        t0 = float(g.pts_s.iloc[0]); 
        nxt = fr.index[fr.index > g.index[-1]]
        t1 = float(fr.pts_s.loc[nxt[0]]) if len(nxt) else float(g.pts_s.iloc[-1] + dt_med)
        dur = t1 - t0
        name = f"{m.video_id}__s{shot}_g{j:02d}.mp4"
        if (m.video_id, shot) in excl: status = "excluded_shot"
        elif dur < CLIP_MIN: status = "too_short"
        else:
            status = "ok"
            subprocess.run(["ffmpeg","-y","-v","error","-i",src,"-ss",f"{t0-st:.4f}","-to",f"{t1-st:.4f}",
                            "-vf","fps=30","-an","-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p",f"{OUT}/{name}"],check=True)
            pr = json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=nb_read_frames,duration","-of","json",f"{OUT}/{name}"],capture_output=True,text=True).stdout)["streams"][0]
        rows.append(dict(video_id=m.video_id, label=m.label, shot_id=shot, seg_idx=j, clip=name if status=="ok" else "",
                         t0_src_s=round(t0,4), t1_src_s=round(t1,4), dur_s=round(dur,3),
                         clip_frames=int(pr["nb_read_frames"]) if status=="ok" else np.nan, status=status))
    print(m.video_id, [ (r["dur_s"], r["status"]) for r in rows if r["video_id"]==m.video_id ], flush=True)
man = pd.DataFrame(rows); man.to_csv("step1_wu/manifest.csv", index=False)
print(man.status.value_counts())
