# 차를 인식하는 (x, y) 좌표를 뽑기 위해 video의 frame을 image로 저장하는 코드

import cv2
import os

# 비디오 파일 목록
video_files = ["partition1_video.mp4", "partition2_video.mp4", "partition3_video.mp4"]

# 입력/출력 경로
video_dir = "./videos"
output_dir = "./images"

os.makedirs(output_dir, exist_ok=True)

# 1분 = 60초
TARGET_TIME = 60

for file in video_files:
    video_path = os.path.join(video_dir, file)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)          # frame per second
    target_frame = int(fps * TARGET_TIME)    # 60초 지점 frame 번호

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()

    if not ret:
        print(f"❌ Failed to capture frame from {file}")
        continue

    # 저장 파일명: partition1_image.png …
    output_name = file.replace("_video.mp4", "_image.png")
    save_path = os.path.join(output_dir, output_name)

    cv2.imwrite(save_path, frame)
    print(f"✅ Saved {save_path}")

    cap.release()
