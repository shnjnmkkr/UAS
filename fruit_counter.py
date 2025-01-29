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

f_boxes=f_boxes[f_boxes[:,5]==1]  # yellow fruits only
b_boxes=b_boxes[b_boxes[:,5]==1]

img=cv2.imread(front)
width=img.shape[1]

for b in b_boxes:
    b[0]=width-b[0]
    b[2]=width-b[2]

# Track which fruits have been matched
matched_front = set()
matched_back = set()

# Find matching pairs
for i, f in enumerate(f_boxes):
    f_center=(f[0]+f[2])/2
    for j, b in enumerate(b_boxes):
        b_center=(b[0]+b[2])/2
        if abs(f_center-b_center)<50:
            matched_front.add(i)
            matched_back.add(j)
            break

overlapping = len(matched_front) 
front_only = len(f_boxes) - len(matched_front)
back_only = len(b_boxes) - len(matched_back)
unique_fruits = overlapping + front_only + back_only

print(f"\nFront view fruits: {len(f_boxes)}")
print(f"Back view fruits: {len(b_boxes)}")
print(f"Fruits visible in both views: {overlapping}")
print(f"Fruits only in front: {front_only}")
print(f"Fruits only in back: {back_only}")
print(f"Total unique fruits: {unique_fruits}")

