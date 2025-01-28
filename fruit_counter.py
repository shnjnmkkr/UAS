import numpy as np
from ultralytics import YOLO
import cv2

model=YOLO('best.pt')
front=r"UAS_DTU_Round_2_Task_data/5/images_close_img_14_23.jpg"
back=r"UAS_DTU_Round_2_Task_data/5/images_close_img_14_23_back.jpg"

f_res=model(front,show=True)
b_res=model(back,show=True)
cv2.waitKey(0)
cv2.destroyAllWindows()

f_boxes=f_res[0].boxes.data.cpu().numpy()
b_boxes=b_res[0].boxes.data.cpu().numpy()
#testing only yellow fruits for now (class 1)
f_boxes=f_boxes[f_boxes[:,5]==1]
b_boxes=b_boxes[b_boxes[:,5]==1]

img=cv2.imread(front)
width=img.shape[1]

both=[]
f_only=[]
b_only=[]

for b in b_boxes:
    b[0]=width-b[0]
    b[2]=width-b[2]

for f in f_boxes:
    f_center=(f[0]+f[2])/2
    matched=False
    
    for b in b_boxes:
        b_center=(b[0]+b[2])/2
        if abs(f_center-b_center)<50:
            both.append(f)
            matched=True
            break
    
    if not matched:
        f_only.append(f)  # fruit only visible in front view
#checking for fruits exclusive to the back view
for b in b_boxes:
    if not any(abs((b[0]+b[2])/2 - (f[0]+f[2])/2)<50 for f in both):
        b_only.append(b)

print(f"\nYellow fruits visible in both views: {len(both)}")
print(f"Yellow fruits only in front: {len(f_only)}")
print(f"Yellow fruits only in back: {len(b_only)}")
print(f"Total unique yellow fruits: {len(both)+len(f_only)+len(b_only)}")

