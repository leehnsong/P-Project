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
# CONFIG (영상이 끊김 없이 움직이도록 설정 변경)
# =====================================================
DEBUG = True
YOLO_W, YOLO_H = 864, 480
FRAME_INTERVAL = 30  # 90에서 30으로 낮춤 (약 1초마다 YOLO 갱신)

model = YOLO("weights/visDrone.pt")
vehicle_ids = [3, 4, 5, 9]

VIDEO_INFO = [
    ("P1", "videos/partition1_video.mp4"),
    ("P2", "videos/partition2_video.mp4"),
    ("P3", "videos/partition3_video.mp4"),
]

PARTITION_SLOTS = {
    "P1": list(range(1, 42)),
    "P2": list(range(42, 67)),
    "P3": list(range(67, 84)),
}
DISABLED_SLOTS = {45, 46, 65, 66}

# 상태 저장소
status_cache = {"P1": {}, "P2": {}, "P3": {}, "last_update": None}
# 각 파티션의 최신 분석 결과(박스 등)를 저장
last_analysis_results = {"P1": (set(), []), "P2": (set(), []), "P3_1": (set(), []), "P3_2": (set(), [])}

# =====================================================
# 시각화 및 분석 함수
# =====================================================
def load_center_slots(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(item["slot"]): (min(item["p1"][0], item["p2"][0]), min(item["p1"][1], item["p2"][1]), 
                                    max(item["p1"][0], item["p2"][0]), max(item["p1"][1], item["p2"][1])) for item in raw}
    except: return {}

CENTER_P1 = load_center_slots("map/partition1_center_slots.json")
CENTER_P2 = load_center_slots("map/partition2_center_slots.json")
CENTER_P3_1 = load_center_slots("map/partition3_1_center_slots.json")
CENTER_P3_2 = load_center_slots("map/partition3_2_center_slots.json")

def draw_realtime(frame, center_rects, occupied_slots, boxes):
    # 원본 프레임에 직접 그리기 (시각화용)
    h, w = frame.shape[:2]
    sx, sy = w / YOLO_W, h / YOLO_H # 박스 좌표를 원본 크기로 복구
    
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), (0, 0, 255), 2)
        
    for slot, (x1, y1, x2, y2) in center_rects.items():
        color = (0, 0, 255) if slot in occupied_slots else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(slot), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# =====================================================
# 백그라운드 워커 (무조건 영상 출력 모드)
# =====================================================
def slot_worker():
    global last_analysis_results
    caps = {name: cv2.VideoCapture(path) for name, path in VIDEO_INFO}
    frame_count = 0

    while True:
        frames = {}
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frames[name] = frame

            # 일정 주기마다 YOLO 분석 수행 (이외의 프레임은 기존 결과 사용)
            if frame_count % FRAME_INTERVAL == 0:
                frame_small = cv2.resize(frame, (YOLO_W, YOLO_H))
                results = model(frame_small, classes=vehicle_ids, conf=0.3, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
                
                # 점유 판단 로직
                def get_occ(rects):
                    occ = set()
                    sx, sy = YOLO_W / frame.shape[1], YOLO_H / frame.shape[0]
                    for bx1, by1, bx2, by2 in boxes:
                        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                        for slot, (rx1, ry1, rx2, ry2) in rects.items():
                            if rx1*sx <= cx <= rx2*sx and ry1*sy <= cy <= ry2*sy:
                                occ.add(slot)
                    return occ, boxes

                if name == "P1": last_analysis_results["P1"] = get_occ(CENTER_P1)
                elif name == "P2": 
                    last_analysis_results["P2"] = get_occ(CENTER_P2)
                    last_analysis_results["P3_1"] = get_occ(CENTER_P3_1)
                elif name == "P3": last_analysis_results["P3_2"] = get_occ(CENTER_P3_2)

        # 데이터 업데이트 (API 응답용)
        merged_occ_p3 = last_analysis_results["P3_1"][0].union(last_analysis_results["P3_2"][0])
        all_occs = {"P1": last_analysis_results["P1"][0], "P2": last_analysis_results["P2"][0], "P3": merged_occ_p3}
        
        for p_name, occ_set in all_occs.items():
            avail = sorted([s for s in PARTITION_SLOTS[p_name] if s not in occ_set])
            status_cache[p_name] = {
                "occupied_slots": sorted(list(occ_set)),
                "available_slots": avail,
                "disabled_slots": sorted([s for s in avail if s in DISABLED_SLOTS]),
                "total_available": len(avail)
            }
        status_cache["last_update"] = time.time()

        # 실시간 화면 출력 (분석 결과가 있든 없든 매 프레임 그리기)
        if DEBUG:
            for name in ["P1", "P2", "P3"]:
                if name in frames:
                    disp = frames[name].copy()
                    if name == "P1":
                        draw_realtime(disp, CENTER_P1, last_analysis_results["P1"][0], last_analysis_results["P1"][1])
                    elif name == "P2":
                        draw_realtime(disp, CENTER_P2, last_analysis_results["P2"][0], last_analysis_results["P2"][1])
                    elif name == "P3":
                        # P3는 두 영상의 분석 결과를 합쳐서 보여줌
                        draw_realtime(disp, CENTER_P3_2, merged_occ_p3, last_analysis_results["P3_2"][1])
                    
                    cv2.imshow(f"Live - {name}", cv2.resize(disp, (640, 360)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_count += 1

threading.Thread(target=slot_worker, daemon=True).start()

@app.get("/status")
def get_status():
    return status_cache