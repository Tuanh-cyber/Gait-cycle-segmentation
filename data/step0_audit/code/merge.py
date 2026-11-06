import pandas as pd, numpy as np, os, glob
H=os.path.expanduser("~/work"); OUT="step0_audit"
m=pd.concat([pd.read_csv(f"{H}/out_c/step0_auto_metrics.csv") if os.path.exists(f"{H}/out_c/step0_auto_metrics.csv") else pd.read_csv(f"{H}/out_c/step0_auto_metrics_part.csv"),
             pd.read_csv(f"{H}/out_f/step0_auto_metrics.csv") if os.path.exists(f"{H}/out_f/step0_auto_metrics.csv") else pd.read_csv(f"{H}/out_f/step0_auto_metrics_part.csv")])
m["video_id"]=m.video.str.replace(r"\.(mp4|mov)$","",regex=True)
sh=pd.read_csv(f"{H}/shots.csv")
# direction auto (recomputed from frame csvs)
rows=[]
for f in sorted(glob.glob(f"{H}/out_*/frames/*_frames.csv")):
    d=pd.read_csv(f); v=os.path.basename(f).replace("_frames.csv",""); u=d[~d.is_dup]
    bg=u.cam_dx.fillna(0); bg=bg.where(bg.abs()<60,0)
    dcx=u.run_cx.interpolate(limit_direction="both").diff().fillna(0); dcx=dcx.where(dcx.abs()<60,0)
    w=dcx-bg; ws=w.sum()/480; fp=(w>0.5).mean(); fn=(w<-0.5).mean()
    strong=abs(ws)>=0.5 and max(fp,fn)>=0.55
    rows.append(dict(video_id=v, dir_auto_score=round(ws,2), dir_auto=int(np.sign(ws)) if strong else 0))
da=pd.DataFrame(rows)
VIS={'C01':0,'C04_1':1,'C04_2':-1,'C04_3':-1,'C09_1':1,'C09_2':-1,'C11':-1,'C12':-1,'C13_1':0,'C13_2':-1,'C17':1,'C18':-1,'C19':1,'C20':1,'C21':-1,'C23_1':1,'C23_2':1,'C23_3':1,'C24':-1,'C25':-1,'C26':-1,'C27':0,'C28_1':1,'C28_2':1,'C28_3':1,'C28_4':1,'C29':-1,'C30':-1,'C31':1,'C38_1':1,'C38_10':-1,'C38_2':0,'C38_3':1,'C38_4':1,'C38_5':1,'C38_6':-1,'C38_7':1,'C38_8':0,'C38_9':0,
 'F02':1,'F0x':1,'F10_1':1,'F10_2':1,'F11':-1,'F12':1,'F17':0,'F18':1,'F19':-1,'F20':1,'F21':1,'F22_1':1,'F22_2':1,'F22_3':1,'F22_4':1,'F23_1':0,'F23_2':1,'F23_3':1,'F24':-1,'F25':-1,'F26':-1,'F27':0,'F28_1':1,'F28_2':0,'F28_3':-1,'F28_4':-1,'F29':-1,'F30':0}
HIRES={'C04_3','C25','F10_2','F25'}  # checked at full resolution
VIEW={'C01':'lateral + cắt cận cảnh bàn chân','C04_1':'lateral','C04_2':'oblique-front','C04_3':'lateral-oblique','C09_1':'lateral (camera tĩnh)','C09_2':'lateral (camera tĩnh)','C11':'oblique-front','C12':'lateral → oblique-front cuối clip','C13_1':'lateral (có frame chuyển cảnh trắng)','C13_2':'lateral','C17':'lateral, khung chặt','C18':'lateral','C19':'lateral','C20':'lateral','C21':'lateral','C23_1':'oblique-front','C23_2':'oblique-front','C23_3':'oblique-front','C24':'lateral, khung chặt','C25':'oblique-front → lateral','C26':'lateral','C27':'lateral','C28_1':'lateral','C28_2':'lateral','C28_3':'lateral','C28_4':'lateral','C29':'oblique-front','C30':'oblique-front','C31':'lateral','C38_1':'lateral','C38_10':'lateral','C38_2':'oblique-rear (chạy ra xa)','C38_3':'oblique-front','C38_4':'lateral','C38_5':'lateral + title card cuối','C38_6':'lateral','C38_7':'lateral','C38_8':'lateral, người chạy rất nhỏ/xa','C38_9':'lateral',
 'F02':'oblique-front','F0x':'lateral → tiến lại gần','F10_1':'oblique-front, đám đông + vật che tiền cảnh','F10_2':'lateral, đám đông','F11':'oblique-front','F12':'lateral','F17':'lateral','F18':'oblique-front','F19':'lateral','F20':'lateral','F21':'lateral, khung chặt','F22_1':'oblique-front','F22_2':'oblique-front','F22_3':'oblique-front','F22_4':'oblique-front','F23_1':'oblique-front','F23_2':'oblique-front','F23_3':'oblique-front','F24':'oblique-front','F25':'oblique-front','F26':'lateral','F27':'lateral','F28_1':'lateral','F28_2':'lateral','F28_3':'lateral','F28_4':'lateral','F29':'oblique-front','F30':'lateral'}
t=m.merge(sh,on="video_id").merge(da,on="video_id")
t["label"]=t.video_id.str[0].map({"C":"CORRECT","F":"INCORRECT"})
t["dir_visual"]=t.video_id.map(VIS)
ef=[]; AF=[]
for v in t.video_id:
    f=glob.glob(f"{H}/out_*/frames/{v}_frames.csv")[0]; d=pd.read_csv(f)
    dd=d.diff_prev.replace([np.inf],np.nan)
    med=np.nanmedian(dd[~d.is_dup]) if (~d.is_dup).sum()>3 else 1.0
    static=(dd<0.25*med).fillna(False).values   # dup OR near-static (animated caption on a still)
    flag=np.zeros(len(d),bool); segs=[]; i=0; n=len(d)
    while i<n:
        if static[i]:
            j=i
            while j+1<n and static[j+1]: j+=1
            t0=d.pts_s.iloc[max(i-1,0)]; t1=d.pts_s.iloc[min(j+1,n-1)]
            if t1-t0>=0.30: flag[i:j+1]=True; segs.append(f"{t0:.2f}-{t1:.2f}")
            i=j+1
        else: i+=1
    d["is_edit_freeze"]=flag; d.to_csv(f,index=False)
    act=~flag; dt=d.pts_s.diff().fillna(0)
    act_dur=float(dt[act].sum()); nu=int(((~d.is_dup)&act).sum())
    AF.append((round(act_dur,2), round(nu/act_dur,2) if act_dur>0 else np.nan))
    ef.append((len(segs), round(sum(float(x.split('-')[1])-float(x.split('-')[0]) for x in segs),2), ";".join(segs)))
t["active_duration_s"]=[a[0] for a in AF]; t["effective_fps_active"]=[a[1] for a in AF]
t["n_edit_freeze"]=[e[0] for e in ef]; t["edit_freeze_total_s"]=[e[1] for e in ef]; t["edit_freeze_segments_s"]=[e[2] for e in ef]
def decide(r):
    a,v=r.dir_auto,r.dir_visual
    if r.video_id in HIRES: return v,"cao","đã kiểm tra full-res"
    if a!=0 and v!=0 and a==v: return v,"cao","auto & visual khớp"
    if a!=0 and v==0: return a,"trung bình","chỉ có auto"
    if a==0 and v!=0: return v,"trung bình","chỉ có visual (auto yếu)"
    if a!=0 and v!=0: return v,"thấp","auto & visual mâu thuẫn"
    return 0,"chưa xác định","cần keypoint"
t[["s_prelim","s_confidence","s_basis"]]=t.apply(lambda r: pd.Series(decide(r)),axis=1)
t["camera_view"]=t.video_id.map(VIEW)
t["mirrored"]="chưa xác minh"
def tier(f): return "≥28 (gần 30 thật)" if f>=28 else "20–28" if f>=20 else "12–20" if f>=12 else "<12 (rất thấp)"
t["fps_tier"]=t.effective_fps_active.map(tier)
def cam(r):
    p=r.pan_path_x_frac
    return "tĩnh" if p<0.1 else "pan nhẹ" if p<0.5 else "pan vừa" if p<1.5 else "pan mạnh"
t["camera_motion"]=t.apply(cam,axis=1)
t["est_strides_rough"]=(np.minimum(t.longest_shot_s,t.active_duration_s)*1.4).round(1)
def flags(r):
    f=[]
    if r.effective_fps_active<12: f.append("effective fps (phần chạy) <12")
    if min(r.longest_shot_s,r.active_duration_s)<2: f.append("clip <2 s (≤ ~2 stride)")
    if r.n_shots>1: f.append(f"{int(r.n_shots)} shot – cắt tại {r.cut_times_s}s")
    if "oblique" in r.camera_view: f.append("không thuần lateral")
    if r.s_confidence in ("thấp","chưa xác định"): f.append("hướng chạy cần keypoint xác nhận")
    if r.pan_path_x_frac>=1.5: f.append("pan mạnh – cần bù chuyển động camera")
    if r.n_edit_freeze>0: f.append(f"freeze biên tập {r.edit_freeze_total_s}s – loại khỏi phân tích")
    if r.effective_fps_active<12 and r.dup_ratio>0.6: f.append("nghi slow-motion bằng nhân frame – Bước 1 (Wu) phải ước lượng tốc độ")
    if "nhỏ" in r.camera_view or "đám đông" in r.camera_view or "vật che" in r.camera_view: f.append("khó tracking")
    return "; ".join(f)
t["flags"]=t.apply(flags,axis=1)
cols=["video_id","label","video","codec","width","height","orientation","container_fps","avg_fps","vfr","duration_s","n_frames","n_unique","n_dup","dup_ratio","effective_fps","effective_fps_active","active_duration_s","fps_tier","unique_dt_median_ms","unique_dt_max_ms","max_freeze_run","n_edit_freeze","edit_freeze_total_s","edit_freeze_segments_s","ambiguous_diff_frames",
 "n_shots","cut_times_s","longest_shot_s","est_strides_rough","camera_view","camera_motion","pan_path_x_frac","pan_total_x_frac","tilt_path_y_frac","zoom_total","cam_valid_ratio",
 "dir_auto_score","dir_auto","dir_visual","s_prelim","s_confidence","s_basis","mirrored","flags"]
t=t[cols].sort_values(["label","video_id"])
t.to_csv(f"{OUT}/step0_metadata.csv",index=False)
readme=pd.DataFrame([
("video_id","Tên file không đuôi"),("label","CORRECT (folder Correct) / INCORRECT (folder False) — tránh chuỗi FALSE bị đọc thành boolean"),
("container_fps / avg_fps","Từ ffprobe; r_frame_rate=60 nhưng avg ~30 ⇒ file VFR"),
("vfr","True nếu khoảng cách PTS giữa các frame không đều (std > 2 ms)"),
("n_unique / n_dup / dup_ratio","Frame trùng: MAD ảnh xám (480px) so với frame trước < min(1.0, 0.12×median MAD của frame thật). Phân bố hai mode rất rõ (trùng <0.3, thật >3)"),
("effective_fps","n_unique / thời lượng thật (theo PTS)"),("effective_fps_active / active_duration_s","Như trên nhưng bỏ các đoạn freeze biên tập — đây là con số dùng cho thiết kế event detection"),
("unique_dt_median/max_ms","Khoảng thời gian thật giữa hai frame unique liên tiếp — dùng làm Δt cho mọi đạo hàm"),
("max_freeze_run","Số frame trùng liên tiếp dài nhất"),
("n_edit_freeze / edit_freeze_*","Đoạn freeze do biên tập: ≥4 frame trùng liên tiếp và kéo dài ≥150 ms (ảnh tĩnh chèn chú thích). Cột is_edit_freeze trong frames CSV. Không được coi là stance"),
("ambiguous_diff_frames","Frame có MAD lửng lơ (có thể là frame blend/nội suy) — nên xem tay nếu >0"),
("n_shots / cut_times_s","Phát hiện cắt cảnh bằng khoảng cách Bhattacharyya histogram HSV > 0.45"),
("est_strides_rough","longest_shot_s × 1.4 stride/s (giả định ~84 stride/phút) — chỉ để ước lượng số cycle khả dụng"),
("camera_view","Đánh giá trực quan trên contact sheet + crop"),
("camera_motion / pan_path_x_frac","Tổng |dx| nền (LK + RANSAC similarity) theo bề rộng khung; tĩnh <0.1, nhẹ <0.5, vừa <1.5, mạnh ≥1.5"),
("zoom_total","Tích hệ số scale của camera qua cả clip (≠1 ⇒ zoom/dolly)"),
("dir_auto_score","Σ(Δx người chạy − Δx nền)/bề rộng; >0 ⇒ chạy sang phải màn hình"),
("dir_auto","±1 khi tín hiệu mạnh (|score|≥0.5 và ≥55% bước cùng dấu), 0 = yếu"),
("dir_visual","Hướng mặt/mũi giày từ crop; 0 = không chắc"),
("s_prelim","Quy ước: +1 = chạy trái→phải màn hình, −1 = phải→trái. Tạm thời — chốt bằng vector heel→toe RTMPose ở Bước 3"),
("mirrored","Không xác định được từ video (overlay text được thêm sau khi lật). Cần xác minh thủ công/nguồn gốc"),
("frames/<video>_frames.csv","Bản đồ từng frame: pts_s, diff_prev, is_dup, unique_id, cam_dx/dy/scale, run_* (bbox chuyển động, thô), hist_dist, is_cut, shot_id"),
],columns=["cột / file","ý nghĩa"])
with pd.ExcelWriter(f"{OUT}/step0_metadata.xlsx",engine="openpyxl") as w:
    t.to_excel(w,sheet_name="metadata",index=False); readme.to_excel(w,sheet_name="README",index=False)
    from openpyxl.styles import Font, PatternFill, Alignment
    ws=w.sheets["metadata"]; ws.freeze_panes="B2"
    for c in ws[1]: c.font=Font(bold=True,color="FFFFFF"); c.fill=PatternFill("solid",fgColor="1F4E78"); c.alignment=Alignment(wrap_text=True,vertical="top")
    for col in ws.columns:
        L=max(len(str(c.value or "")) for c in col); ws.column_dimensions[col[0].column_letter].width=min(max(9,L+2),45)
    red=PatternFill("solid",fgColor="F8D7DA"); yel=PatternFill("solid",fgColor="FFF3CD")
    hdr=[c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        r={h:c for h,c in zip(hdr,row)}
        if r["effective_fps_active"].value<12: r["effective_fps_active"].fill=red
        elif r["effective_fps_active"].value<20: r["effective_fps_active"].fill=yel
        if r["s_confidence"].value in ("thấp","chưa xác định"): r["s_confidence"].fill=red
        if r["longest_shot_s"].value<2: r["longest_shot_s"].fill=yel
    wr=w.sheets["README"]; wr.column_dimensions["A"].width=34; wr.column_dimensions["B"].width=110
print(t[["video_id","effective_fps","effective_fps_active","active_duration_s","fps_tier","n_shots","camera_motion","s_prelim","s_confidence"]].to_string())
print(t.s_confidence.value_counts()); print(t.fps_tier.value_counts()); print(t.camera_motion.value_counts()); print(t.vfr.value_counts())
