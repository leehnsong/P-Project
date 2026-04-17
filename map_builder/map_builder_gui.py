import cv2
import numpy as np
import json
import math

# ================= 설정 =================
# 도면 전체 크기 (가로 393, 세로 852) (처음에 만들었던 map의 크기와 동일)
CANVAS_WIDTH = 393
CANVAS_HEIGHT = 852

# 저장될 JSON 파일 이름
OUTPUT_JSON_PATH = "custom_rotated_slots.json"

# 전역 상태 변수
slots = []           # 슬롯 정보 저장 [{cx, cy, w, h, angle}]
selected_idx = -1    # 현재 선택된 슬롯의 인덱스
dragging = False     # 드래그 상태
# ========================================

def get_rotated_rect_points(cx, cy, w, h, angle_deg):
    """중심점, 너비, 높이, 각도를 받아 회전된 사각형의 4개 꼭짓점 좌표를 반환"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    hw, hh = w / 2, h / 2

    # 중심점 기준 4개 코너
    corners = [
        (-hw, -hh),  # 좌상
        (hw, -hh),   # 우상
        (hw, hh),    # 우하
        (-hw, hh)    # 좌하
    ]

    rotated_corners = []
    for x, y in corners:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        rotated_corners.append([int(cx + rx), int(cy + ry)])

    return np.array(rotated_corners, dtype=np.int32)

def mouse_callback(event, x, y, flags, param):
    global slots, selected_idx, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 위치가 기존 슬롯 내부에 있는지 확인 (가장 위에 그려진 것부터 확인)
        clicked_idx = -1
        for i in range(len(slots)-1, -1, -1):
            pts = get_rotated_rect_points(slots[i]['cx'], slots[i]['cy'], slots[i]['w'], slots[i]['h'], slots[i]['angle'])
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                clicked_idx = i
                break

        if clicked_idx != -1:
            # 기존 슬롯 선택 및 드래그 시작
            selected_idx = clicked_idx
            dragging = True
        else:
            # 빈 공간 클릭 시 새로운 슬롯 생성
            new_slot = {"cx": x, "cy": y, "w": 50, "h": 80, "angle": 0}
            slots.append(new_slot)
            selected_idx = len(slots) - 1
            dragging = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # 드래그 중이면 선택된 슬롯의 중심 좌표 업데이트
        if dragging and selected_idx != -1:
            slots[selected_idx]['cx'] = x
            slots[selected_idx]['cy'] = y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def on_trackbar(val):
    # 트랙바 콜백용 (메인 루프에서 값을 읽어오므로 패스)
    pass

def main():
    global slots, selected_idx

    cv2.namedWindow("Map Builder")
    cv2.setMouseCallback("Map Builder", mouse_callback)

    # 상단 트랙바(슬라이더) 생성
    cv2.createTrackbar("Angle", "Map Builder", 0, 360, on_trackbar)
    cv2.createTrackbar("Width", "Map Builder", 50, 200, on_trackbar)
    cv2.createTrackbar("Height", "Map Builder", 80, 200, on_trackbar)

    print("\n" + "="*50)
    print("🚗 OpenCV 회전 주차장 맵 빌더 🚗")
    print("- 빈 공간 [클릭]: 새 주차칸 생성")
    print("- 주차칸 [클릭 & 드래그]: 이동")
    print("- 상단 [슬라이더]: 선택된 주차칸 크기/각도 조절")
    print("- 'd' 키: 선택된 주차칸 삭제")
    print("- 's' 키: JSON 파일로 저장")
    print("- 'q' 키: 종료")
    print("="*50 + "\n")

    last_selected_idx = -1

    while True:
        # 1. 배경 생성 (393 x 852 크기의 어두운 회색)
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 60, dtype=np.uint8)

        # 2. 선택된 슬롯이 바뀌면 트랙바 UI 값을 해당 슬롯 값으로 동기화
        if selected_idx != last_selected_idx:
            if selected_idx != -1:
                cv2.setTrackbarPos("Angle", "Map Builder", slots[selected_idx]['angle'])
                cv2.setTrackbarPos("Width", "Map Builder", slots[selected_idx]['w'])
                cv2.setTrackbarPos("Height", "Map Builder", slots[selected_idx]['h'])
            last_selected_idx = selected_idx

        # 3. 선택된 슬롯이 있으면 트랙바에서 값을 읽어와 실시간 적용
        if selected_idx != -1:
            slots[selected_idx]['angle'] = cv2.getTrackbarPos("Angle", "Map Builder")
            slots[selected_idx]['w'] = max(10, cv2.getTrackbarPos("Width", "Map Builder")) # 최소 10
            slots[selected_idx]['h'] = max(10, cv2.getTrackbarPos("Height", "Map Builder"))

        # 4. 모든 슬롯 화면에 그리기
        for i, slot in enumerate(slots):
            pts = get_rotated_rect_points(slot['cx'], slot['cy'], slot['w'], slot['h'], slot['angle'])
            
            # 주차칸은 흰색으로 채우기
            cv2.fillPoly(canvas, [pts], (220, 220, 220))
            
            # 테두리 그리기 (선택된 슬롯은 초록색으로 하이라이트, 나머지는 검정색)
            color = (0, 255, 0) if i == selected_idx else (30, 30, 30)
            thickness = 2 if i == selected_idx else 1
            cv2.polylines(canvas, [pts], True, color, thickness)
            
            # 번호 표시
            cv2.putText(canvas, str(i+1), (slot['cx'] - 5, slot['cy'] + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 5. 화면 출력
        cv2.imshow("Map Builder", canvas)
        
        # 6. 키보드 이벤트 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # 종료
            break
        elif key == ord('d'):  # 선택된 슬롯 삭제
            if selected_idx != -1:
                del slots[selected_idx]
                selected_idx = -1
                last_selected_idx = -1
                print("삭제 완료")
        elif key == ord('s'):  # JSON 저장
            # 회전된 사각형의 4개 점(points)을 모두 저장하도록 포맷 구성
            output_data = []
            for i, slot in enumerate(slots):
                pts = get_rotated_rect_points(slot['cx'], slot['cy'], slot['w'], slot['h'], slot['angle'])
                output_data.append({
                    "slot": i + 1,
                    "points": pts.tolist()  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                })
            
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)
            print(f"🎉 {OUTPUT_JSON_PATH} 에 총 {len(output_data)}개의 슬롯이 저장되었습니다!")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()