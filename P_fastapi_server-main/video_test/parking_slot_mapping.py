# map의 (x1, y1), (x2, y2) 좌표와 주차칸 번호 mapping
# partition1에는 41개의 parking slot
# partition2에는 25개의 parking slot (4개는 장애인 주차 구역)
# partition3에는 17개의 parking slot 
# 총 83개의 parking slot 존재 (장애인 주차 구역은 4개)
# 1번부터 83번까지 주차 구역에 숫자 부여. 

# parking_slot_mapping.py

# partition1: 41 slots
# partition2: 25 slots (42~66, 이 중 4개는 장애인: 42~45)
# partition3: 17 slots (67~83)
# total: 83 slots

import cv2
import json
import numpy as np

# ===================== 설정 =====================

PARTITIONS = [
    {
        "name": "partition1",
        "image": "map/partition1_map.png",
        "start_slot": 1,
        "num_slots": 41,
    },
    {
        "name": "partition2",
        "image": "map/partition2_map.png",
        "start_slot": 42,
        "num_slots": 25,
    },
    {
        "name": "partition3",
        "image": "map/partition3_map.png",
        "start_slot": 67,
        "num_slots": 17,
    },
]

# 장애인 주차 구역
DISABLED_SLOTS = {45, 46, 65, 66}

# 너무 작은 노이즈 박스를 걸러내기 위한 최소 면적 (필요하면 조절)
MIN_AREA = 300

# 결과가 들어갈 딕셔너리
mapping_parking_slot = {p["name"]: [] for p in PARTITIONS}

# =================================================


def detect_slots(image):
    """
    흰색 슬롯(직사각형/대각선 포함)을 자동으로 찾아서
    각 슬롯을 꼭짓점 4개로 표현한 리스트를 반환.
    return: [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ... ]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 흰색 박스만 남도록 threshold (배경이 어두운 회색, 슬롯이 밝은 회색이라고 가정)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        if len(cnt) < 4:
            continue

        rect = cv2.minAreaRect(cnt)          # (center, (w,h), angle)
        (w, h) = rect[1]
        area = w * h
        if area < MIN_AREA:
            continue

        box = cv2.boxPoints(rect)            # 4개 꼭짓점 (float)
        box = np.int32(box)                  # int로 변환
        poly = box.tolist()
        polys.append(poly)

    return polys


def label_partition(partition_config):
    """
    한 partition(이미지)에 대해:
    - 슬롯들을 자동으로 탐지 (4점 polygon)
    - 사용자가 슬롯 안을 클릭하면, 그 polygon에
      현재 slot 번호를 부여하고 mapping에 저장
    """
    name = partition_config["name"]
    path = partition_config["image"]
    start_slot = partition_config["start_slot"]
    num_slots = partition_config["num_slots"]

    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] 이미지를 읽을 수 없음: {path}")
        return

    window_name = f"{name} - click slots in order ({start_slot}~{start_slot + num_slots - 1})"

    # 자동 슬롯 탐지
    polys = detect_slots(img)
    if not polys:
        print(f"[WARN] {name}: 슬롯을 찾지 못했습니다.")
        return

    n_slots_found = len(polys)
    print(f"[INFO] {name}: 자동으로 찾은 슬롯 개수 = {n_slots_found}")

    # 상태 관리용 배열
    assigned = [False] * len(polys)              # 이 polygon이 이미 번호가 매겨졌는지
    assigned_disabled = [False] * len(polys)     # 이 polygon이 disabled 슬롯인지
    assigned_slot_number = [None] * len(polys)   # 이 polygon에 어떤 slot 번호가 붙었는지

    state = {
        "current_slot": start_slot,
        "end_slot": start_slot + num_slots - 1,
        "img": img,
        "vis": img.copy(),
        "polys": polys,
        "assigned": assigned,
        "assigned_disabled": assigned_disabled,
        "assigned_slot_number": assigned_slot_number,
        "name": name,
    }

    def redraw():
        """전체를 다시 그림: 배경 + 모든 슬롯(배정된/안 된)"""
        vis = state["img"].copy()

        for i, poly in enumerate(state["polys"]):
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

            if state["assigned"][i]:
                disabled = state["assigned_disabled"][i]
                slot_no = state["assigned_slot_number"][i]
                color = (0, 255, 255) if disabled else (0, 255, 0)  # disabled면 노란색
                cv2.polylines(vis, [pts], True, color, 2)

                x_text, y_text = poly[0]  # 첫 점 근처에 번호 표시
                cv2.putText(
                    vis,
                    str(slot_no),
                    (x_text, y_text - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                # 아직 번호 안 붙은 슬롯은 파란색
                cv2.polylines(vis, [pts], True, (255, 0, 0), 1)

        state["vis"] = vis

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        cur = state["current_slot"]
        end = state["end_slot"]
        if cur > end:
            print(f"[{name}] 이미 모든 슬롯 번호를 지정했습니다.")
            return

        click_point = (float(x), float(y))

        # 어떤 polygon 안에 클릭했는지 찾기
        clicked_index = None
        for i, poly in enumerate(state["polys"]):
            if state["assigned"][i]:
                continue
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            inside = cv2.pointPolygonTest(pts, click_point, False)
            if inside >= 0:  # 안 또는 경계
                clicked_index = i
                break

        if clicked_index is None:
            print(f"[{name}] 슬롯 영역 밖을 클릭했습니다. 다시 클릭하세요.")
            return

        slot_number = cur
        disabled = slot_number in DISABLED_SLOTS

        # 이 polygon에 slot 번호 할당
        state["assigned"][clicked_index] = True
        state["assigned_disabled"][clicked_index] = disabled
        state["assigned_slot_number"][clicked_index] = slot_number

        # 전역 mapping에도 추가
        poly_points = state["polys"][clicked_index]
        mapping_parking_slot[name].append(
            {
                "slot": slot_number,
                "points": poly_points,          # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                "disabled": disabled,
            }
        )

        print(
            f"[{name}] slot {slot_number} 저장 (disabled={disabled}): "
            f"{poly_points}"
        )

        state["current_slot"] += 1
        redraw()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    # 처음 한 번 전체 그림
    redraw()

    print(f"\n[{name}]")
    print(
        f"화면에 표시된 슬롯들 중에서, {start_slot}번부터 {start_slot + num_slots - 1}번까지 "
        "번호 순서대로 슬롯 안을 한 번씩 클릭하세요."
    )
    print("이 partition에서 작업을 끝내고 다음으로 넘어가려면 'n' 키를 누르세요.")
    print("이 partition 작업을 처음부터 다시 하고 싶으면 'r' 키를 누르세요.\n")

    while True:
        cv2.imshow(window_name, state["vis"])
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            # 이 partition 전체 리셋
            print(f"[{name}] 이 partition을 리셋합니다.")
            mapping_parking_slot[name] = []
            for i in range(len(state["polys"])):
                state["assigned"][i] = False
                state["assigned_disabled"][i] = False
                state["assigned_slot_number"][i] = None
            state["current_slot"] = start_slot
            redraw()

        if key == ord("n"):
            # 이 partition 작업 종료
            break

    cv2.destroyWindow(window_name)


def main():
    for part in PARTITIONS:
        label_partition(part)

    # 각 partition 별로 slot 번호 기준으로 정렬(안 해도 되지만 혹시 몰라서)
    for name in mapping_parking_slot:
        mapping_parking_slot[name].sort(key=lambda x: x["slot"])

    # ===== 결과 저장 =====
    # 1) JSON
    with open("map/mapping_parking_slot.json", "w", encoding="utf-8") as f:
        json.dump(mapping_parking_slot, f, indent=2, ensure_ascii=False)

    # 2) Python 파일
    with open("map/mapping_parking_slot.py", "w", encoding="utf-8") as f:
        f.write("mapping_parking_slot = ")
        f.write(json.dumps(mapping_parking_slot, indent=2, ensure_ascii=False))
        f.write("\n")

    print("\n✅ 저장 완료: mapping_parking_slot.json, mapping_parking_slot.py")


if __name__ == "__main__":
    main()
