import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math
import urllib.request
import os

# 下载模型文件（只在第一次运行时下载）
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("正在下载手部追踪模型...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("模型下载完成")

# 初始化 HandLandmarker
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# 手部连接关系（用于画骨架）
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def hand_center(landmarks, w, h):
    lm = landmarks[9]
    return (int(lm.x * w), int(lm.y * h))

def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 200), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

cap = cv2.VideoCapture(0)
print("摄像头启动，按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # 转成 MediaPipe Image 格式
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    hand_expand_amount = 0.0
    centers = []

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            draw_hand(frame, landmarks, w, h)
            center = hand_center(landmarks, w, h)
            centers.append(center)
            cv2.circle(frame, center, 8, (0, 255, 200), -1)

        if len(centers) == 2:
            dist = distance(centers[0], centers[1])
            hand_expand_amount = min(dist / w, 1.0)
            cv2.line(frame, centers[0], centers[1], (0, 200, 255), 2)
            mid = (
                (centers[0][0] + centers[1][0]) // 2,
                (centers[0][1] + centers[1][1]) // 2
            )
            cv2.putText(frame, f"expand: {hand_expand_amount:.2f}",
                        mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    print(f"双手数量: {len(centers)}  |  hand_expand_amount: {hand_expand_amount:.3f}")

    cv2.putText(frame, f"hands: {len(centers)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"expand: {hand_expand_amount:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("Please Perform More Clearly", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("退出")