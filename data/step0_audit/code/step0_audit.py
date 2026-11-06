"""Step 0 audit: freeze map, true timestamps, effective fps, camera pan,
running direction, cadence sanity check. One CSV per video + summary."""
import subprocess, json, sys, os, numpy as np, pandas as pd, cv2

W_AN = 480  # analysis width

def probe(path):
    s = json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of","json",path],capture_output=True,text=True).stdout)["streams"][0]
    ts = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "frame=best_effort_timestamp_time","-of","csv=p=0",path],capture_output=True,text=True).stdout.split()
    ts = np.array([float(t.strip(',')) for t in ts if t.strip(',') not in ("","N/A")])
    return s, ts

def frames(path, W, H, w=W_AN):
    h = int(round(H*w/W/2)*2)
    raw = subprocess.run(["ffmpeg","-v","error","-i",path,"-vsync","passthrough",
        "-vf",f"scale={w}:{h}:flags=area,format=gray","-f","rawvideo","pipe:"],capture_output=True).stdout
    return np.frombuffer(raw,np.uint8).reshape(-1,h,w)

def cam_motion(a, b):
    """similarity transform a->b from background features (RANSAC)."""
    p0 = cv2.goodFeaturesToTrack(a, 400, 0.01, 8)
    if p0 is None or len(p0) < 20: return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21,21), maxLevel=3)
    ok = st.ravel()==1
    if ok.sum() < 20: return None, 0
    M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None: return None, 0
    return M, inl.mean()

def analyse(path):
    s, ts = probe(path)
    W, H = s["width"], s["height"]
    F = frames(path, W, H)
    n = len(F)
    if len(ts) != n:  # fallback
        ts = ts[:n] if len(ts) > n else np.arange(n)/eval(s["avg_frame_rate"])
    order = np.argsort(ts, kind="stable"); ts = ts[order]  # decode order == display for passthrough; keep sorted
    Ff = F.astype(np.float32)
    d = np.r_[np.inf, np.abs(np.diff(Ff,axis=0)).mean(axis=(1,2))]
    hi = d[(d>1.0)&np.isfinite(d)]
    hi_med = np.median(hi) if len(hi) else 1.0
    thr = min(1.0, 0.12*hi_med)
    dup = d < thr
    ambiguous = int(((d>=thr)&(d<0.35*hi_med)).sum())
    uid = np.cumsum(~dup)-1
    uniq_idx = np.where(~dup)[0]
    t_u = ts[uniq_idx]
    dt_u = np.diff(t_u)
    dur = ts[-1]-ts[0] + (np.median(np.diff(ts)) if n>1 else 0)
    # longest freeze run
    runs, r = [], 0
    for x in dup:
        r = r+1 if x else 0; runs.append(r)
    max_freeze = int(max(runs)) if runs else 0

    # camera + runner on unique frames
    h_an, w_an = F.shape[1:]
    rec = {k: np.full(n, np.nan) for k in ["cam_dx","cam_dy","cam_scale","cam_inlier","run_cx","run_cy","run_x0","run_x1","run_y0","run_y1"]}
    kernel = np.ones((5,5),np.uint8)
    for a_i, b_i in zip(uniq_idx[:-1], uniq_idx[1:]):
        a, b = F[a_i], F[b_i]
        M, inl = cam_motion(a, b)
        if M is None: continue
        sc = np.sqrt(abs(np.linalg.det(M[:,:2])))
        rec["cam_dx"][b_i], rec["cam_dy"][b_i], rec["cam_scale"][b_i], rec["cam_inlier"][b_i] = M[0,2], M[1,2], sc, inl
        aw = cv2.warpAffine(a, M, (w_an,h_an), borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.absdiff(aw, b)
        diff[:4,:]=0; diff[-4:,:]=0; diff[:,:4]=0; diff[:,-4:]=0
        m = (cv2.GaussianBlur(diff,(5,5),0) > 25).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.dilate(m, kernel, iterations=3)
        k, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        if k <= 1: continue
        j = 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        if stats[j,cv2.CC_STAT_AREA] < 0.002*w_an*h_an: continue
        x,y,ww,hh,_ = stats[j]
        rec["run_cx"][b_i], rec["run_cy"][b_i] = cent[j]
        rec["run_x0"][b_i],rec["run_x1"][b_i],rec["run_y0"][b_i],rec["run_y1"][b_i] = x,x+ww,y,y+hh

    dx = np.nan_to_num(rec["cam_dx"]); cum_bg = np.cumsum(dx)
    v = np.isfinite(rec["run_cx"])
    out = dict(video=os.path.basename(path), folder=os.path.basename(os.path.dirname(path)),
        label="CORRECT" if "Correct" in path else "FALSE",
        codec=s["codec_name"], width=W, height=H, orientation="portrait" if H>W else "landscape",
        container_fps=round(eval(s["r_frame_rate"]),3), avg_fps=round(eval(s["avg_frame_rate"]),3),
        n_frames=n, duration_s=round(dur,3),
        vfr=bool(np.std(np.diff(ts))>0.002) if n>2 else False,
        n_unique=int((~dup).sum()), n_dup=int(dup.sum()), dup_ratio=round(dup.mean(),3),
        effective_fps=round((~dup).sum()/dur,2),
        unique_dt_median_ms=round(1000*np.median(dt_u),1) if len(dt_u) else np.nan,
        unique_dt_max_ms=round(1000*np.max(dt_u),1) if len(dt_u) else np.nan,
        max_freeze_run=max_freeze, ambiguous_diff_frames=ambiguous, dup_threshold=round(thr,3))
    # camera pan
    fin = np.isfinite(rec["cam_dx"])
    out["cam_valid_ratio"] = round(fin[uniq_idx[1:]].mean(),3) if len(uniq_idx)>1 else 0
    out["pan_total_x_frac"] = round(abs(cum_bg[-1])/w_an,3)
    out["pan_path_x_frac"] = round(np.nansum(np.abs(rec["cam_dx"]))/w_an,3)
    out["tilt_path_y_frac"] = round(np.nansum(np.abs(rec["cam_dy"]))/h_an,3)
    out["zoom_total"] = round(float(np.nanprod(rec["cam_scale"])),3)
    # runner world displacement relative to background
    if v.sum() >= 8:
        tv = ts[v]; xw = rec["run_cx"][v] - cum_bg[v]; ximg = rec["run_cx"][v]
        bh = np.nanpercentile((rec["run_y1"]-rec["run_y0"])[v], 75)
        sl = np.polyfit(tv, xw, 1)[0]; sl_img = np.polyfit(tv, ximg, 1)[0]
        out["runner_bbox_h_frac"] = round(bh/h_an,3)
        out["runner_speed_bh_per_s"] = round(sl/bh,2)   # signed, + = moving to image right
        out["runner_img_drift_frac"] = round((ximg[-1]-ximg[0])/w_an,3) if len(ximg)>1 else np.nan
        # bbox size trend (approaching/receding => non-lateral)
        hh = (rec["run_y1"]-rec["run_y0"])[v]
        out["bbox_h_trend_ratio"] = round(np.polyval(np.polyfit(tv,hh,1),tv[-1])/max(1,np.polyval(np.polyfit(tv,hh,1),tv[0])),2)
        # cadence: periodicity of runner vertical centroid + mask height on uniform time grid
        g = np.arange(tv[0], tv[-1], 1/60)
        if len(g) > 90:
            sig = np.interp(g, tv, (rec["run_x1"]-rec["run_x0"])[v])  # bbox width oscillates with leg spread
            sig = sig - np.convolve(sig, np.ones(30)/30, mode="same")
            sig = sig[15:-15]
            f = np.fft.rfftfreq(len(sig), 1/60); P = np.abs(np.fft.rfft(sig*np.hanning(len(sig))))**2
            band = (f>0.8)&(f<5.0)
            if band.any():
                fp = f[band][np.argmax(P[band])]
                out["step_freq_hz"] = round(fp,2); out["cadence_spm"] = round(60*fp,0)
    else:
        out["runner_speed_bh_per_s"] = np.nan
    fr = pd.DataFrame(dict(frame=np.arange(n), pts_s=ts, diff_prev=d, is_dup=dup, unique_id=uid, **rec))
    return out, fr

if __name__ == "__main__":
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(f"{outdir}/frames", exist_ok=True)
    vids = sorted([f"{root}/{d}/{f}" for d in ["Correct","False"] for f in os.listdir(f"{root}/{d}") if f.lower().endswith((".mp4",".mov"))])
    if len(sys.argv) > 3: vids = [v for v in vids if any(k in v for k in sys.argv[3:])]
    rows = []
    for p in vids:
        try:
            o, fr = analyse(p); rows.append(o)
            fr.to_csv(f"{outdir}/frames/{os.path.splitext(o['video'])[0]}_frames.csv", index=False)
            print(o["video"], o["effective_fps"], o.get("runner_speed_bh_per_s"), o.get("cadence_spm"), o["pan_path_x_frac"], flush=True)
        except Exception as e:
            print("ERR", p, e, flush=True)
    pd.DataFrame(rows).to_csv(f"{outdir}/step0_auto_metrics{'_part' if len(sys.argv)>3 else ''}.csv", index=False)
