import subprocess, json, numpy as np, pandas as pd, cv2, glob, os, sys
res={}
for v in sorted(glob.glob("Correct/*.mp4")+glob.glob("False/*.m*")):
    name=os.path.splitext(os.path.basename(v))[0]
    fr=pd.read_csv(glob.glob(os.path.expanduser(f"~/work/out_*/frames/{name}_frames.csv"))[0])
    s=json.loads(subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height","-of","json",v],capture_output=True,text=True).stdout)["streams"][0]
    w=160; h=int(round(s["height"]*w/s["width"]/2)*2)
    raw=subprocess.run(["ffmpeg","-v","error","-i",v,"-vsync","passthrough","-vf",f"scale={w}:{h},format=bgr24","-f","rawvideo","pipe:"],capture_output=True).stdout
    F=np.frombuffer(raw,np.uint8).reshape(-1,h,w,3)[:len(fr)]
    hs=[]; flat=[]
    for f in F:
        hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV); H=cv2.calcHist([hsv],[0,1],None,[24,16],[0,180,0,256]); cv2.normalize(H,H); hs.append(H)
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); flat.append(g.std())
    dist=np.r_[0,[cv2.compareHist(hs[i-1],hs[i],cv2.HISTCMP_BHATTACHARYYA) for i in range(1,len(hs))]]
    flat=np.array(flat)
    cut=(dist>0.45)
    card=flat<18   # near-uniform frames: title cards / blank / fades
    fr["hist_dist"]=dist; fr["is_cut"]=cut; fr["is_card"]=card
    # shot id
    fr["shot_id"]=np.cumsum(cut)
    fr.to_csv(glob.glob(os.path.expanduser(f"~/work/out_*/frames/{name}_frames.csv"))[0],index=False)
    lens=fr.groupby("shot_id").pts_s.agg(lambda x:x.max()-x.min())
    res[name]=dict(n_cuts=int(cut.sum()), n_shots=int(cut.sum()+1), longest_shot_s=round(lens.max(),2), card_frames=int(card.sum()), cut_times_s=";".join(f"{t:.2f}" for t in fr.pts_s[cut]))
    print(name,res[name],flush=True)
pd.DataFrame(res).T.rename_axis("video_id").to_csv(os.path.expanduser("~/work/shots.csv"))
