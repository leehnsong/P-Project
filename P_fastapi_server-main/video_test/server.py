import cv2
import json
import numpy as np
import threading
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CONFIG
# =====================================================
DEBUG = True  # ← 영상 디버그 모드 ON/OFF

YOLO_W, YOLO_H = 854, 480
model = YOLO("weights/visDrone.pt")
vehicle_ids = [3, 4, 5, 9]  # car, van, truck, motor

VIDEO_P1 = "videos/partition1_video.mp4"
VIDEO_P2 = "videos/partition2_video.mp4"
VIDEO_P3 = "videos/partition3_video.mp4"

VIDEO_INFO = [
    ("P1", VIDEO_P1),
    ("P2", VIDEO_P2),
    ("P3", VIDEO_P3),
]

active_caps = {}

PARTITION_SLOTS = {
    "P1": list(range(1, 42)),
    "P2": list(range(42, 72)),
    "P3": list(range(73, 84)),
}

DISABLED_SLOTS = {45, 46, 65, 66}

# =====================================================
# 디버그 프레임을 Partition별로 따로 저장
# =====================================================
DEBUG_FRAMES = {
    "P1": None,
    "P2": None,
    "P3": None,
}

# =====================================================
# JSON 로더
# =====================================================
def load_center_slots(path):
    with open(path, "r") as f:
        raw = json.load(f)

    out = {}
    for item in raw:
        slot = int(item["slot"])
        x1, y1 = item["p1"]
        x2, y2 = item["p2"]
        out[slot] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return out


CENTER_P1 = load_center_slots("map/partition1_center_slots.json")
CENTER_P2 = load_center_slots("map/partition2_center_slots.json")
CENTER_P3 = load_center_slots("map/partition3_center_slots.json")


# =====================================================
# 디버그 시각화 함수
# =====================================================
def draw_debug_frame(frame_small, scaled_rects, boxes, occupied_slots):
    debug = frame_small.copy()

    # YOLO 박스 (빨강)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # 슬롯 rect (초록=빈칸, 빨강=점유)
    for slot, (sx1, sy1, sx2, sy2) in scaled_rects.items():
        color = (0, 255, 0)
        if slot in occupied_slots:
            color = (0, 0, 255)
        cv2.rectangle(debug, (int(sx1), int(sy1)), (int(sx2), int(sy2)), color, 2)
        cv2.putText(debug, str(slot), (int(sx1), int(sy1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return debug


# =====================================================
# 점유 판단 함수
# =====================================================
def detect_slot_occupied(frame, center_slots, part_name):
    if frame is None:
        return set()

    H, W = frame.shape[:2]
    frame_small = cv2.resize(frame, (YOLO_W, YOLO_H))

    sx = YOLO_W / W
    sy = YOLO_H / H

    scaled = {
        slot: (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        for slot, (x1, y1, x2, y2) in center_slots.items()
    }

    result = model(frame_small, classes=vehicle_ids, conf=0.4, verbose=False)[0]

    boxes = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()

    occupied = set()

    # 중심점 기준 점유 판단
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for slot, (sx1, sy1, sx2, sy2) in scaled.items():
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                occupied.add(slot)
                break

    # ---- Partition별 Debug Frame 저장 ----
    if DEBUG:
        DEBUG_FRAMES[part_name] = draw_debug_frame(frame_small, scaled, boxes, occupied)

    return occupied


# =====================================================
# 상태 캐시
# =====================================================
status_cache = {
    "P1": None,
    "P2": None,
    "P3": None,
    "last_update": None
}


# =====================================================
# 백그라운드 YOLO 워커 (멀티 디버그 창 적용)
# =====================================================
def slot_worker():
    while True:
        for name, cap in active_caps.items():

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if name == "P1":
                occ = detect_slot_occupied(frame, CENTER_P1, "P1")
            elif name == "P2":
                occ = detect_slot_occupied(frame, CENTER_P2, "P2")
            elif name == "P3":
                occ = detect_slot_occupied(frame, CENTER_P3, "P3")

            total = PARTITION_SLOTS[name]
            occupied = sorted(list(occ))
            available = sorted([s for s in total if s not in occ])
            disabled = sorted([s for s in total if s in DISABLED_SLOTS])

            status_cache[name] = {
                "occupied_slots": occupied,
                "available_slots": available,
                "disabled_slots": disabled,
                "total_available": len(available)
            }

        status_cache["last_update"] = time.time()

        # ----- 각 파티션마다 디버그 창 따로 띄우기 -----
        if DEBUG:
            for part in ["P1", "P2", "P3"]:
                frame_debug = DEBUG_FRAMES[part]
                if frame_debug is not None:
                    cv2.imshow(f"YOLO Debug - {part}", frame_debug)

            cv2.waitKey(1)

        time.sleep(0.05)


# =====================================================
# 서버 시작 시 비디오 로드
# =====================================================
def init_server():
    for name, path in VIDEO_INFO:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            print(f"✔ 활성화된 비디오: {name}")
            active_caps[name] = cap
        else:
            print(f"✖ 비디오 없음: {path}")

    th = threading.Thread(target=slot_worker, daemon=True)
    th.start()

init_server()


# =====================================================
# API
# =====================================================
@app.get("/status")
def get_status():
    return status_cache
