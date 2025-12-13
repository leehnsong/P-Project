import cv2
import json

# partition2 이미지 불러오기
img = cv2.imread("detected_images/partition2_detected.png")
orig = img.copy()

drawing = False
ix, iy = -1, -1

slot_start_number = 42   # partition2 시작 번호
slot_index = slot_start_number

slots = []  # 저장되는 슬롯 좌표 리스트


def redraw_all_boxes():
    """Undo 후 전체 슬롯 박스를 다시 그리는 함수"""
    global img
    img = orig.copy()
    for item in slots:
        p1 = item["p1"]
        p2 = item["p2"]
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, f"{item['slot']}", (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, img, slot_index, slots

    # 클릭 → 시작점
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # 드래그 중
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        redraw_all_boxes()
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.putText(img, f"Slot {slot_index}", (ix, iy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 드래그 끝 → 저장
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        x1, y1 = ix, iy
        x2, y2 = x, y

        p1 = (min(x1, x2), min(y1, y2))
        p2 = (max(x1, x2), max(y1, y2))

        slots.append({
            "slot": slot_index,
            "p1": p1,
            "p2": p2
        })

        print(f"[Saved] Slot {slot_index}: p1={p1}, p2={p2}")

        redraw_all_boxes()
        slot_index += 1


cv2.namedWindow("partition2-slot-labeling")
cv2.setMouseCallback("partition2-slot-labeling", draw_rect)

print("=== partition2 슬롯 표시 ===")
print("왼쪽 드래그 → 영역 생성")
print("u → Undo (직전 슬롯 제거)")
print("q → 저장 후 종료\n")

while True:
    cv2.imshow("partition2-slot-labeling", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('u'):
        if len(slots) > 0:
            removed = slots.pop()
            print(f"[Undo] 슬롯 {removed['slot']} 삭제됨")
            slot_index -= 1
            redraw_all_boxes()
        else:
            print("[Undo] 삭제할 슬롯 없음")


cv2.destroyAllWindows()

# JSON 저장
with open("map/partition2_center_slots.json", "w", encoding="utf-8") as f:
    json.dump(slots, f, indent=4)

print("\n✔ partition2 좌표 저장 완료 → map/partition2_center_slots.json")
