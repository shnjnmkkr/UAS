import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

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

def get_relative_pos(fruit, box):
    box_width = abs(box[2] - box[0])
    box_left = min(box[0], box[2])
    fruit_center = (fruit[0] + fruit[2]) / 2
    return (fruit_center - box_left) / box_width

for pair_num, (front, back) in enumerate(image_pairs, 1):
    print(f"\nProcessing image pair {pair_num}:")
    
    # Get detections
    f_res = model(front, show=True)
    b_res = model(back, show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    f_boxes = f_res[0].boxes.data.cpu().numpy()
    b_boxes = b_res[0].boxes.data.cpu().numpy()

    # Get and sort white boxes
    f_white = f_boxes[f_boxes[:,5]==3]
    b_white = b_boxes[b_boxes[:,5]==3]
    f_white = f_white[np.argsort(f_white[:,0])]      # left to right
    b_white = b_white[np.argsort(-b_white[:,0])]     # right to left

    # Mirror back view coordinates
    width = cv2.imread(front).shape[1]
    for b in b_boxes:
        x1, x2 = b[0], b[2]
        b[0] = width - x2
        b[2] = width - x1

    # Process each fruit color
    for class_id in range(3):
        f_fruits = f_boxes[f_boxes[:,5]==class_id]
        b_fruits = b_boxes[b_boxes[:,5]==class_id]
        matched_front = set()
        matched_back = set()

        # Match fruits in corresponding white boxes
        for fw, bw in zip(f_white, b_white):
            # Get fruits in each box
            f_in_box = [(i, get_relative_pos(f, fw)) 
                       for i, f in enumerate(f_fruits) 
                       if fw[0] <= f[0] and f[2] <= fw[2]]
            
            b_in_box = [(j, get_relative_pos(b, bw)) 
                       for j, b in enumerate(b_fruits) 
                       if bw[0] <= b[0] and b[2] <= bw[2]]

            # Match fruits with similar positions
            for i, f_pos in f_in_box:
                for j, b_pos in b_in_box:
                    if abs(f_pos - b_pos) < 0.021:
                        matched_front.add(i)
                        matched_back.add(j)
                        break

        # Count results
        both = len(matched_front)
        front_only = len(f_fruits) - both
        back_only = len(b_fruits) - len(matched_back)
        total = both + front_only + back_only

        color = model.names[class_id]
        print(f"\n{color.upper()} fruits:")
        print(f"Front view: {len(f_fruits)}")
        print(f"Back view: {len(b_fruits)}")
        print(f"Visible in both views: {both}")
        print(f"Only in front: {front_only}")
        print(f"Only in back: {back_only}")
        print(f"Total no. of fruits: {total}")

