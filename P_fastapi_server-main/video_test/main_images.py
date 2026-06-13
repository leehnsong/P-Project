import cv2
import json
import numpy as np
from ultralytics import YOLO

# ======================= 설정 =======================

# YOLO 모델
model = YOLO("weights/visDrone.pt")
vehicle_ids = [3, 4, 5, 9]  # car, van, truck, motor

# 파티션 슬롯 개수
partition1_slots = 41
partition2_slots = 25
partition3_slots = 17

DISABLED_SLOTS = {45, 46, 65, 66}

# 이미지 파일
IMG_P1 = "images/partition1_image.png"
IMG_P2 = "images/partition2_image.png"
IMG_P3 = "images/partition3_image.png"

# 슬롯 JSON
CENTER_JSON_P1 = "map/partition1_center_slots.json"
CENTER_JSON_P2 = "map/partition2_center_slots.json"
CENTER_JSON_P3_1 = "map/partition3_1_center_slots.json"  # 67~72 from p2 view
CENTER_JSON_P3_2 = "map/partition3_2_center_slots.json"  # 73~83 from p3 view

# mapping JSON + map images
MAPPING_JSON = "map/mapping_parking_slot.json"
MAP_P1 = "map/partition1_map.png"
MAP_P2 = "map/partition2_map.png"
MAP_P3 = "map/partition3_map.png"

# ==================================================


def load_center_slots(json_path):
    """p1/p2 사각형 기반 center slot JSON 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    center_slots = {}
    for item in data:
        slot = int(item["slot"])
        x1, y1 = item["p1"]
        x2, y2 = item["p2"]

        # 정렬
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        center_slots[slot] = (left, top, right, bottom)

    return center_slots


def load_mapping_slots(json_path):
    """map에 색칠하기 위한 4점 polygon 가져오기"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for part_name, slots in data.items():
        mapping[part_name] = {}
        for item in slots:
            slot = int(item["slot"])
            pts = np.array(item["points"], dtype=np.int32)
            disabled = bool(item.get("disabled", False))
            mapping[part_name][slot] = {
                "points": pts,
                "disabled": disabled,
            }
    return mapping


def detect_occupied_slots(img, center_rects, conf=0.4):
    """
    차량 bounding box 중심이 center_rects 영역에 포함되면 점유
    center_rects: {slot: (x1, y1, x2, y2)}
    """
    occupied = set()

    if img is None:
        return occupied

    results = model(img, classes=vehicle_ids, conf=conf, verbose=False)
    det = results[0]

    if det.boxes is None:
        return occupied

    boxes = det.boxes.xyxy.cpu().numpy()

    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for slot, (sx1, sy1, sx2, sy2) in center_rects.items():
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                occupied.add(slot)
                break

    return occupied


def draw_partition_map(base_img, mapping_slot_dict, occupied_slots, partition_name=""):
    """
    mapping_parking_slot.json 기반 polygon 영역 색칠
    """
    vis = base_img.copy()

    for slot, info in mapping_slot_dict.items():
        poly = info["points"].reshape((-1, 1, 2))
        disabled = info["disabled"]

        if slot in occupied_slots:
            color = (0, 0, 255)  # 빨강
        else:
            if disabled:
                color = (0, 255, 255)  # 노랑
            else:
                color = (0, 255, 0)  # 초록

        cv2.fillPoly(vis, [poly], color)
        cv2.polylines(vis, [poly], True, (0, 0, 0), 1)

        x_text, y_text = info["points"][0]
        cv2.putText(vis, str(slot), (int(x_text), int(y_text) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if partition_name:
        cv2.putText(vis, partition_name, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def main():
    # ---------------- JSON 로드 ----------------
    center_p1 = load_center_slots(CENTER_JSON_P1)      # 1~41
    center_p2 = load_center_slots(CENTER_JSON_P2)      # 42~66
    center_p3_1 = load_center_slots(CENTER_JSON_P3_1)  # 67~72
    center_p3_2 = load_center_slots(CENTER_JSON_P3_2)  # 73~83

    mapping_slots = load_mapping_slots(MAPPING_JSON)

    # ---------------- 이미지 로드 ----------------
    img1 = cv2.imread(IMG_P1)
    img2 = cv2.imread(IMG_P2)
    img3 = cv2.imread(IMG_P3)

    map1 = cv2.imread(MAP_P1)
    map2 = cv2.imread(MAP_P2)
    map3 = cv2.imread(MAP_P3)

    # ---------------- 차량 탐지 ----------------
    occ_p1 = detect_occupied_slots(img1, center_p1)
    occ_p2 = detect_occupied_slots(img2, center_p2)
    occ_p3 = detect_occupied_slots(img2, center_p3_1) | detect_occupied_slots(img3, center_p3_2)

    # ---------------- 잔여 계산 ----------------
    p1_available = partition1_slots - len(occ_p1)
    p2_available = partition2_slots - len(occ_p2)
    p3_available = partition3_slots - len(occ_p3)

    print("=== 이미지 기반 주차 점유 결과 ===")
    print(f"Partition1: occupied={len(occ_p1)}, available={p1_available}")
    print(f"Partition2: occupied={len(occ_p2)}, available={p2_available}")
    print(f"Partition3: occupied={len(occ_p3)}, available={p3_available}")

    # ---------------- 맵 색칠 ----------------
    map1_vis = draw_partition_map(map1, mapping_slots["partition1"], occ_p1,
                                  partition_name=f"P1 avail: {p1_available}")

    map2_vis = draw_partition_map(map2, mapping_slots["partition2"], occ_p2,
                                  partition_name=f"P2 avail: {p2_available}")

    map3_vis = draw_partition_map(map3, mapping_slots["partition3"], occ_p3,
                                  partition_name=f"P3 avail: {p3_available}")

    # ---------------- 결과 출력 ----------------
    cv2.imshow("Partition1 Image Detection", img1)
    cv2.imshow("Partition2 Image Detection", img2)
    cv2.imshow("Partition3 Image Detection", img3)

    cv2.imshow("Partition1 Map", map1_vis)
    cv2.imshow("Partition2 Map", map2_vis)
    cv2.imshow("Partition3 Map", map3_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
