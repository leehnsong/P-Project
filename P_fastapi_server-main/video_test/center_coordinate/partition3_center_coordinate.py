import cv2
import json

def label_slots_for_image(img_path, window_name, start_slot, end_slot):
    """
    img_path 이미지에서 start_slot ~ end_slot까지
    마우스로 드래그해서 슬롯 박스를 찍고,
    slots 리스트를 반환한다.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    orig = img.copy()

    drawing = False
    ix, iy = -1, -1
    slot_index = start_slot
    slots_phase = []  # 이 이미지에서 찍은 슬롯들만 저장

    def redraw_all():
        nonlocal img
        img = orig.copy()
        for item in slots_phase:
            p1 = tuple(item["p1"])
            p2 = tuple(item["p2"])
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                img,
                str(item["slot"]),
                (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, slot_index, slots_phase, img

        # 시작점
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        # 드래그 미리보기
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            redraw_all()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"Slot {slot_index}",
                (ix, iy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # 드래그 끝 → 슬롯 하나 확정
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

            x1, y1 = ix, iy
            x2, y2 = x, y

            p1 = (min(x1, x2), min(y1, y2))
            p2 = (max(x1, x2), max(y1, y2))

            slots_phase.append({
                "slot": slot_index,
                "p1": p1,
                "p2": p2
            })

            print(f"[Saved] Slot {slot_index}: p1={p1}, p2={p2}")
            redraw_all()
            slot_index += 1

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_cb)

    print(f"\n=== {window_name} ===")
    print(f"이미지: {img_path}")
    print(f"슬롯 번호: {start_slot} ~ {end_slot}")
    print("왼쪽 드래그 → 슬롯 박스 생성")
    print("u 키 → Undo (직전 슬롯 삭제)")
    print("q 키 → 다음 단계로 진행\n")

    while True:
        # 모든 슬롯 다 찍었으면 자동 종료
        if slot_index > end_slot:
            cv2.imshow(window_name, img)
            key = cv2.waitKey(500)
            if key == ord('q'):
                break
            continue

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == ord('u'):
            if len(slots_phase) > 0:
                removed = slots_phase.pop()
                slot_index -= 1
                print(f"[Undo] 슬롯 {removed['slot']} 삭제됨")
                redraw_all()
            else:
                print("[Undo] 삭제할 슬롯 없음")

    cv2.destroyWindow(window_name)
    return slots_phase


if __name__ == "__main__":
    all_slots = []

    # 1) partition2_detected.png 에서 67~72까지 라벨링
    slots_p2 = label_slots_for_image(
        img_path="detected_images/partition2_detected.png",
        window_name="partition2: slots 67-72",
        start_slot=67,
        end_slot=72,
    )
    all_slots.extend(slots_p2)

    # 2) partition3_detected.png 에서 73~83까지 라벨링
    slots_p3 = label_slots_for_image(
        img_path="detected_images/partition3_detected.png",
        window_name="partition3: slots 73-83",
        start_slot=73,
        end_slot=83,
    )
    all_slots.extend(slots_p3)

    # 한 JSON으로 저장 (partition3 전체 슬롯이라는 의미로 파일 이름만 이렇게)
    with open("map/partition3_center_slots.json", "w", encoding="utf-8") as f:
        json.dump(all_slots, f, indent=4)

    print("\n✔ 슬롯 67~83 좌표 저장 완료 → map/partition3_center_slots.json")
