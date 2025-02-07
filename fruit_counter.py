import numpy as np
from ultralytics import YOLO
import cv2

model=YOLO('best.pt')

image_pairs = [
    (r"UAS_DTU_Round_2_Task_data/1/images_close_img_3_0.jpg",
     r"UAS_DTU_Round_2_Task_data/1/images_close_img_3_0_back.jpg"),
    (r"UAS_DTU_Round_2_Task_data/2/images_close_img_7_4.jpg",
     r"UAS_DTU_Round_2_Task_data/2/images_close_img_7_20.jpg"),
    (r"UAS_DTU_Round_2_Task_data/3/images_close_img_11_6.jpg",
     r"UAS_DTU_Round_2_Task_data/3/images_close_img_11_24.jpg"),
    (r"UAS_DTU_Round_2_Task_data/4/images_close_img_12_0.jpg",
     r"UAS_DTU_Round_2_Task_data/4/images_close_img_12_9.jpg"),
    (r"UAS_DTU_Round_2_Task_data/5/images_close_img_14_23.jpg", 
     r"UAS_DTU_Round_2_Task_data/5/images_close_img_14_23_back.jpg"),
    (r"UAS_DTU_Round_2_Task_data/6/images_close_img_21_2.jpg",
     r"UAS_DTU_Round_2_Task_data/6/images_close_img_21_6.jpg"),
    (r"UAS_DTU_Round_2_Task_data/7/imageuptodown_img_4_1.jpg",
     r"UAS_DTU_Round_2_Task_data/7/imageuptodown_img_4_5.jpg"),
    (r"UAS_DTU_Round_2_Task_data/8/imageuptodown_img_7_4.jpg",
     r"UAS_DTU_Round_2_Task_data/8/imageuptodown_img_7_7.jpg"),
]

for pair_num, (front, back) in enumerate(image_pairs, 1):
    print(f"\nProcessing image pair {pair_num}:")
    
    f_res=model(front,show=True)
    b_res=model(back,show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    f_boxes=f_res[0].boxes.data.cpu().numpy()
    b_boxes=b_res[0].boxes.data.cpu().numpy()

    for class_id in range(3):
        f_boxes_class=f_boxes[f_boxes[:,5]==class_id]
        b_boxes_class=b_boxes[b_boxes[:,5]==class_id]

        img=cv2.imread(front)
        width=img.shape[1]

        for b in b_boxes_class:
            b[0]=width-b[0]
            b[2]=width-b[2]

        matched_front = set()
        matched_back = set()

        for i, f in enumerate(f_boxes_class):
            f_center=(f[0]+f[2])/2
            for j, b in enumerate(b_boxes_class):
                b_center=(b[0]+b[2])/2
                if abs(f_center-b_center)<50:
                    matched_front.add(i)
                    matched_back.add(j)
                    break

        overlapping = len(matched_front) 
        front_only = len(f_boxes_class) - len(matched_front)
        back_only = len(b_boxes_class) - len(matched_back)
        unique_fruits = overlapping + front_only + back_only

        color = model.names[class_id]
        print(f"\n{color.upper()} fruits:")
        print(f"Front view: {len(f_boxes_class)}")
        print(f"Back view: {len(b_boxes_class)}")
        print(f"Visible in both views: {overlapping}")
        print(f"Only in front: {front_only}")
        print(f"Only in back: {back_only}")
        print(f"Total unique: {unique_fruits}")

