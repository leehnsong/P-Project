# 각 칸마다 object detection된 bounding box의 중심 좌표가 해당 칸 안에 있으면 주차된 것으로 간주


import cv2
import json

# 이미지 불러오기
img = cv2.imread("detected_images/partition1_detected.png")
orig = img.copy()

drawing = False
ix, iy = -1, -1
slot_index = 1

slots = []  # 저장되는 좌표들


def redraw_all_boxes():
    """Undo 후 전체 박스를 다시 그리는 함수"""
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

    # 왼쪽 버튼 눌렀을 때 → 시작점 기록
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # 드래그 중일 때 → 실시간 박스 미리보기
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        redraw_all_boxes()
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.putText(img, f"Slot {slot_index}", (ix, iy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 마우스 버튼 떼면 → 박스 확정 저장
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

        # 화면에 그리기
        redraw_all_boxes()

        slot_index += 1


cv2.namedWindow("partition1-slot-labeling")
cv2.setMouseCallback("partition1-slot-labeling", draw_rect)

print("=== 드래그로 슬롯 박스 지정 ===")
print("왼쪽 드래그 → 영역 저장")
print("u 키 → 직전 슬롯 무르기 (Undo)")
print("q 키 → 종료 및 파일 저장\n")

while True:
    cv2.imshow("partition1-slot-labeling", img)
    key = cv2.waitKey(1)

    # 종료
    if key == ord('q'):
        break

    # Undo 기능
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
with open("map/partition1_center_slots.json", "w", encoding="utf-8") as f:
    json.dump(slots, f, indent=4)

print("\n✔ 모든 좌표 저장 완료! → map/partition1_center_slots.json")
