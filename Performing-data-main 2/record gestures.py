import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
DATA_PATH = "gesture_data.pkl"
WINDOW_NAME = "gesture recorder - click state / SAVE / ESC"
RECORD_EVERY_FRAMES = 3

LABEL_NAMES = {
    0: "STABLE / quiet hands",
    1: "EXPAND / open hands",
    2: "INTERPRETED / clear sign",
    3: "OVERINTERPRETED / complex motion",
    4: "UNCLASSIFIED / strange or blocked",
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

COLORS = {
    0: (90, 220, 180),
    1: (255, 180, 80),
    2: (90, 190, 255),
    3: (220, 90, 255),
    4: (255, 70, 70),
}

BUTTONS = {}
SAVE_BUTTON = None
CLEAR_BUTTON = None


def hand_scale(landmarks):
    wrist = landmarks[0]
    middle_base = landmarks[9]
    scale = ((middle_base.x - wrist.x) ** 2 + (middle_base.y - wrist.y) ** 2) ** 0.5
    return max(scale, 0.03)


def extract_features(result):
    left = [0.0] * 42
    right = [0.0] * 42

    if not result.hand_landmarks:
        return np.array(left + right)

    for idx, landmarks in enumerate(result.hand_landmarks[:2]):
        side = result.handedness[idx][0].category_name
        wrist = landmarks[0]
        scale = hand_scale(landmarks)
        features = []

        for lm in landmarks:
            features.extend([
                (lm.x - wrist.x) / scale,
                (lm.y - wrist.y) / scale,
            ])

        if side == "Left":
            left = features
        else:
            right = features

    return np.array(left + right)


def load_data():
    if not os.path.exists(DATA_PATH):
        return [], []
    with open(DATA_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved.get("X", []), saved.get("y", [])


def save_data(samples_x, samples_y):
    with open(DATA_PATH, "wb") as f:
        pickle.dump({"X": samples_x, "y": samples_y}, f)


def counts(samples_y):
    return {label: samples_y.count(label) for label in range(5)}


def draw_text(frame, text, x, y, size=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)


def draw_hands(frame, result):
    if result is None or not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        color = COLORS.get(hand_index, (255, 255, 255))
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


def inside(rect, x, y):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(frame, rect, label, color, active=False):
    x1, y1, x2, y2 = rect
    fill = tuple(int(c * (0.55 if active else 0.22)) for c in color)
    border = color if active else tuple(int(c * 0.75) for c in color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2 if active else 1, cv2.LINE_AA)
    draw_text(frame, label, x1 + 10, y1 + 28, 0.52, (255, 255, 255), 1)


def make_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Put it next to this script.")
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    global BUTTONS, SAVE_BUTTON, CLEAR_BUTTON

    samples_x, samples_y = load_data()
    detector = make_detector()
    camera = cv2.VideoCapture(0)
    state = {
        "active_label": None,
        "message": "Click STATE 0-4 to start/stop recording. Click SAVE, press ESC to quit.",
        "last_record_frame": -999,
        "frame_count": 0,
        "running": True,
    }

    print("Gesture recorder opened.")
    print("Click STATE 0-4 to start continuous recording. Click the same state again to stop.")
    print("Record about 25-30 samples per state. Single-hand and two-hand samples are both accepted.")

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for label, rect in BUTTONS.items():
            if inside(rect, x, y):
                if state["active_label"] == label:
                    state["active_label"] = None
                    state["message"] = f"Stopped STATE {label}."
                    print(state["message"])
                else:
                    state["active_label"] = label
                    state["last_record_frame"] = -999
                    state["message"] = f"Recording STATE {label}: {LABEL_NAMES[label]}"
                    print(state["message"])
                return
        if SAVE_BUTTON and inside(SAVE_BUTTON, x, y):
            save_data(samples_x, samples_y)
            state["message"] = f"Saved {len(samples_x)} samples to {DATA_PATH}"
            print(state["message"])
            return
        if CLEAR_BUTTON and inside(CLEAR_BUTTON, x, y):
            samples_x.clear()
            samples_y.clear()
            save_data(samples_x, samples_y)
            state["active_label"] = None
            state["message"] = "Cleared all gesture samples."
            print(state["message"])
            return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while state["running"]:
        ok, frame = camera.read()
        if not ok:
            break

        state["frame_count"] += 1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        hand_count = len(result.hand_landmarks) if result is not None and result.hand_landmarks else 0
        has_hands = hand_count > 0
        has_two_hands = hand_count >= 2

        dark = cv2.addWeighted(frame, 0.55, np.zeros_like(frame), 0.45, 0)
        draw_hands(dark, result)
        h, w = dark.shape[:2]

        # Button layout is recalculated every frame so resizing still works.
        BUTTONS = {}
        button_w = max(112, min(150, (w - 56) // 5 - 8))
        button_h = 44
        top = 112
        left = 28
        gap = 8
        for label in range(5):
            x1 = left + label * (button_w + gap)
            BUTTONS[label] = (x1, top, x1 + button_w, top + button_h)
        SAVE_BUTTON = (28, h - 84, 130, h - 42)
        CLEAR_BUTTON = (144, h - 84, 250, h - 42)

        c = counts(samples_y)
        draw_text(dark, "GESTURE RECORDER", 28, 38, 0.8, (255, 255, 255), 1)
        draw_text(dark, "Click a STATE to start continuous recording. Click it again to stop.", 28, 70, 0.5, (230, 230, 230), 1)
        draw_text(dark, f"samples  0:{c[0]}  1:{c[1]}  2:{c[2]}  3:{c[3]}  4:{c[4]}", 28, 98, 0.55, (255, 230, 160), 1)

        for label, rect in BUTTONS.items():
            draw_button(dark, rect, f"STATE {label}", COLORS[label], state["active_label"] == label)

        hand_color = (120, 255, 160) if has_two_hands else ((90, 210, 255) if has_hands else (100, 100, 255))
        draw_text(dark, f"HANDS DETECTED: {hand_count}/2", 28, top + button_h + 32, 0.62, hand_color, 1)
        if state["active_label"] is None:
            draw_text(dark, "RECORDING: OFF", 28, top + button_h + 62, 0.62, (210, 210, 210), 1)
        else:
            draw_text(dark, f"RECORDING: STATE {state['active_label']}  move slightly", 28, top + button_h + 62, 0.62, COLORS[state["active_label"]], 1)

        if not has_hands:
            draw_text(dark, "NO HAND DETECTED - recording waits until a hand appears", 28, top + button_h + 94, 0.58, (90, 90, 255), 1)
        elif not has_two_hands:
            draw_text(dark, "ONE HAND OK - record both single-hand and two-hand examples", 28, top + button_h + 94, 0.55, (90, 210, 255), 1)

        if state["active_label"] is not None and has_hands:
            if state["frame_count"] - state["last_record_frame"] >= RECORD_EVERY_FRAMES:
                label = state["active_label"]
                samples_x.append(extract_features(result).copy())
                samples_y.append(label)
                state["last_record_frame"] = state["frame_count"]
                state["message"] = f"REC state {label} with {hand_count} hand(s), count={samples_y.count(label)}"
                print(state["message"])

        draw_button(dark, SAVE_BUTTON, "SAVE", (120, 255, 160), False)
        draw_button(dark, CLEAR_BUTTON, "CLEAR", (255, 90, 90), False)
        draw_text(dark, "ESC/q quit   keys 0-4 also toggle states   s save   x clear", 270, h - 56, 0.48, (215, 215, 215), 1)
        draw_text(dark, state["message"], 270, h - 26, 0.48, (150, 235, 255) if has_hands else (100, 100, 255), 1)

        cv2.imshow(WINDOW_NAME, dark)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord("s"):
            save_data(samples_x, samples_y)
            state["message"] = f"Saved {len(samples_x)} samples to {DATA_PATH}"
            print(state["message"])
        elif key == ord("x"):
            samples_x.clear()
            samples_y.clear()
            save_data(samples_x, samples_y)
            state["active_label"] = None
            state["message"] = "Cleared all gesture samples."
            print(state["message"])
        elif key in [ord(str(i)) for i in range(5)]:
            label = int(chr(key))
            if state["active_label"] == label:
                state["active_label"] = None
                state["message"] = f"Stopped STATE {label}."
            else:
                state["active_label"] = label
                state["last_record_frame"] = -999
                state["message"] = f"Recording STATE {label}: {LABEL_NAMES[label]}"
            print(state["message"])

    camera.release()
    cv2.destroyAllWindows()
    save_data(samples_x, samples_y)
    print(f"Finished. Saved {len(samples_x)} samples to {DATA_PATH}.")


if __name__ == "__main__":
    main()
