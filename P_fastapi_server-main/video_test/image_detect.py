# image detection해서 bounding box가 그려진 이미지를 detected_images 폴더에 저장하는 코드


import cv2
import numpy as np
# from parking_slot_mapping import mapping_parking_slot
from ultralytics import YOLO
import os

# 1) VisDrone용 YOLOv8 모델 로드
model = YOLO("weights/visDrone.pt")
print(model.names)

# {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
# 차량 계열 클래스만 탐지 (car, motorcycle, truck)
vehicle_ids = [3, 4, 5, 9]  # car, van, truck, motor


# ----------------------------
# 1) 이미지 목록 정의
# ----------------------------
image_list = {
    "partition1": "images/partition1_image.png",
    "partition2": "images/partition2_image.png",
    "partition3": "images/partition3_image.png",
}

# 저장 폴더 생성
os.makedirs("detected_images", exist_ok=True)

# ----------------------------
# 2) 반복 처리
# ----------------------------
for name, path in image_list.items():

    print(f"\n====== [{name}] 이미지 처리 중 ======")

    img = cv2.imread(path)
    if img is None:
        print(f"❌ 이미지 로드 실패: {path}")
        continue

    # YOLO 탐지
    results = model(img, classes=vehicle_ids, conf=0.4)

    # 탐지된 bbox 그리기
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            # Rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

            print(f"[det-{name}] {cls_name:>8s} conf={conf:.2f}, bbox=({x1}, {y1}, {x2}, {y2})")

    # ----------------------------
    # 3) 결과 저장
    # ----------------------------
    save_path = f"detected_images/{name}_detected.png"
    cv2.imwrite(save_path, img)
    print(f"✔ 저장 완료 → {save_path}")

print("\n🎉 모든 이미지 처리 완료!")
