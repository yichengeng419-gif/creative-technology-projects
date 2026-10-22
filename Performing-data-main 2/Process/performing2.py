import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math
import os
import numpy as np
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
import pickle
from pythonosc import udp_client

# ── OSC 输出 ──────────────────────────────────────────────
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 7000)

# ── 模型下载 ──────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("正在下载手部追踪模型...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("模型下载完成")

# ── MediaPipe 初始化 ──────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

STATE_NAMES = {
    0: "STABLE",
    1: "EXPAND",
    2: "INTERPRETED",
    3: "OVERINTERPRETED",
    4: "ERROR"
}

STATE_COLORS = {
    0: (100, 220, 180),
    1: (100, 180, 255),
    2: (180, 100, 255),
    3: (50, 180, 255),
    4: (60, 80, 220)
}

# ── 特征提取 ──────────────────────────────────────────────
def extract_features(hand_landmarks_list, w, h):
    all_features = []
    for landmarks in hand_landmarks_list[:2]:
        wrist = landmarks[0]
        wx, wy = wrist.x, wrist.y
        feats = []
        for lm in landmarks:
            feats.append(lm.x - wx)
            feats.append(lm.y - wy)
        all_features.extend(feats)
    while len(all_features) < 84:
        all_features.append(0.0)
    return np.array(all_features[:84])

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

# ── 数据 & 分类器 ─────────────────────────────────────────
DATA_PATH = "gesture_data.pkl"
samples_X = []
samples_y = []
knn = None
recording_state = None
record_frames = []
RECORD_SECONDS = 3
FPS_ESTIMATE = 30
RECORD_LIMIT = RECORD_SECONDS * FPS_ESTIMATE

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        saved = pickle.load(f)
        samples_X = saved["X"]
        samples_y = saved["y"]
    print(f"已加载已有数据：{len(samples_X)} 条样本")
    if len(samples_X) >= 5:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(samples_X, samples_y)
        print("分类器已就绪")

# ── 主循环 ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("\n── 操作说明 ──────────────────────────")
print("按 0-4：录制对应手势（每个 3 秒）")
print("按 T  ：手动训练分类器")
print("按 R  ：清除所有数据重新录制")
print("按 Q  ：退出")
print("──────────────────────────────────────\n")

gesture_state = -1
state_history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    hand_expand_amount = 0.0
    centers = []
    features = None

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

        features = extract_features(result.hand_landmarks, w, h)

    # ── 录制逻辑 ──────────────────────────────────────────
    if recording_state is not None and features is not None:
        record_frames.append(features)
        progress = len(record_frames) / RECORD_LIMIT
        bar_w = int(w * progress)
        cv2.rectangle(frame, (0, h-20), (bar_w, h), (0, 220, 120), -1)
        cv2.putText(frame,
                    f"录制 {STATE_NAMES[recording_state]}... {len(record_frames)}/{RECORD_LIMIT}",
                    (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)

        if len(record_frames) >= RECORD_LIMIT:
            samples_X.extend(record_frames)
            samples_y.extend([recording_state] * len(record_frames))
            print(f"✓ 手势 {recording_state} ({STATE_NAMES[recording_state]}) 录制完成，{len(record_frames)} 帧")
            with open(DATA_PATH, "wb") as f:
                pickle.dump({"X": samples_X, "y": samples_y}, f)
            recording_state = None
            record_frames = []

            recorded = set(samples_y)
            if len(recorded) == 5:
                print("5 个手势全部录制完成，正在训练分类器...")
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(samples_X, samples_y)
                print("✓ 分类器训练完成，开始实时分类")

    # ── 实时分类 + OSC 输出 ───────────────────────────────
    if knn is not None and features is not None and recording_state is None:
        pred = knn.predict([features])[0]
        state_history.append(pred)
        gesture_state = max(set(state_history), key=list(state_history).count)

        name = STATE_NAMES.get(gesture_state, "?")
        color = STATE_COLORS.get(gesture_state, (255, 255, 255))
        cv2.putText(frame, f"STATE: {gesture_state} {name}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        # 发送 OSC
        osc_client.send_message("/gesture_state", float(gesture_state))
        osc_client.send_message("/hand_expand", float(hand_expand_amount))

        print(f"state: {gesture_state} {name}  |  expand: {hand_expand_amount:.3f}")
    else:
        recorded = set(samples_y)
        missing = [STATE_NAMES[i] for i in range(5) if i not in recorded]
        if missing:
            cv2.putText(frame, f"待录制: {', '.join(missing)}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── 键盘控制 ──────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
        s = int(chr(key))
        if recording_state is None:
            recording_state = s
            record_frames = []
            print(f"开始录制手势 {s}: {STATE_NAMES[s]}，请做动作...")

    elif key == ord('t') or key == ord('T'):
        if len(samples_X) > 0:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(samples_X, samples_y)
            print(f"✓ 手动训练完成，共 {len(samples_X)} 条样本")
        else:
            print("没有数据，请先录制手势")

    elif key == ord('r') or key == ord('R'):
        samples_X, samples_y = [], []
        knn = None
        gesture_state = -1
        if os.path.exists(DATA_PATH):
            os.remove(DATA_PATH)
        print("已清除所有数据，重新开始")

    elif key == ord('q') or key == ord('Q'):
        break

    cv2.imshow("Please Perform More Clearly", frame)

cap.release()
cv2.destroyAllWindows()
print("退出")