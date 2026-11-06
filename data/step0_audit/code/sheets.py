import cv2, numpy as np, os, sys, glob
root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
vids = sorted(glob.glob(f"{root}/Correct/*")+glob.glob(f"{root}/False/*"))
vids = [v for v in vids if v.lower().endswith((".mp4",".mov"))]
H=200; rows=[]
for v in vids:
    cap=cv2.VideoCapture(v); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ims=[]
    for k in np.linspace(0.08,0.92,5):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(k*n)); ok,f=cap.read()
        if not ok: f=np.zeros((H,H,3),np.uint8)
        f=cv2.resize(f,(int(f.shape[1]*H/f.shape[0]),H)); ims.append(f)
    row=np.hstack([np.pad(i,((0,0),(0,4),(0,0))) for i in ims])
    lab=np.full((H,150,3),255,np.uint8); cv2.putText(lab,os.path.basename(v)[:-4],(5,105),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,200),2)
    rows.append(np.hstack([lab,row]))
Wmax=max(r.shape[1] for r in rows)
rows=[np.pad(r,((0,6),(0,Wmax-r.shape[1]),(0,0)),constant_values=230) for r in rows]
for i in range(0,len(rows),6):
    cv2.imwrite(f"{out}/sheet_{i//6:02d}.jpg", np.vstack(rows[i:i+6]), [cv2.IMWRITE_JPEG_QUALITY,80])
print(len(rows))
