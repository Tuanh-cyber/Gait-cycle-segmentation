import cv2
from rtmlib import BodyWithFeet

model = BodyWithFeet(mode="balanced", backend="onnxruntime", device="cpu")
cap = cv2.VideoCapture("data/Correct/C09_1.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ok, img = cap.read()
kp, sc = model(img)
print("Số người phát hiện:", len(kp))
print("Shape keypoint:", kp.shape)
print("Heel trái/phải x:", kp[0][24][0], kp[0][25][0])
print("Big toe trái/phải x:", kp[0][20][0], kp[0][21][0])