import os
import math
import pickle
import queue
import threading
import random
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from dorothy import Dorothy
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

try:
    from diffusion_utils import get_device, load_pipe
except ImportError:
    get_device = None
    load_pipe = None


WIDTH, HEIGHT = 1280, 720
MODEL_PATH = "hand_landmarker.task"
DATA_PATH = "gesture_data.pkl"
SD_SIZE = 256
SD_ENABLED = True
SD_MIN_FRAMES_BETWEEN_JOBS = 120
SD_STEPS = 5
SD_GUIDANCE = 1.15

STATE_NAMES = {
    0: "STABLE",
    1: "EXPAND",
    2: "INTERPRETED",
    3: "OVERINTERPRETED",
    4: "BODY CANNOT BE CLASSIFIED",
}

STATE_TEXT = {
    0: "STABLE HAND PATTERN",
    1: "EXPANSION DETECTED",
    2: "PATTERN FOUND",
    3: "PLEASE GESTURE MORE CLEARLY",
    4: "BODY CANNOT BE CLASSIFIED",
}

STATE_COLORS = {
    0: (90, 220, 180),
    1: (255, 180, 80),
    2: (90, 190, 255),
    3: (220, 90, 255),
    4: (255, 70, 70),
}

SD_PROMPTS = {
    0: (
        "calm biometric thermal scan, cold blue skin temperature map, "
        "soft medical camera noise, subtle contour lines, quiet diagnostic texture"
    ),
    1: (
        "expanding orange thermal body scan, warm infrared skin glow, "
        "breathing heat map, soft blooming edges, layered sensor aura"
    ),
    2: (
        "cyan machine vision diagnostic scan, hand tracking traces, "
        "transparent data texture, spectral skin analysis, precise medical overlay"
    ),
    3: (
        "purple unstable thermal scan, corrupted gesture data, glitching skin map, "
        "fragmented sensor trails, overinterpreted machine vision"
    ),
    4: (
        "red emergency body scan, unclassified human signal, broken thermal camera, "
        "dense error texture, damaged diagnostic image, warning heat map"
    ),
}

SD_STRENGTH = {
    0: 0.30,
    1: 0.40,
    2: 0.46,
    3: 0.56,
    4: 0.62,
}

THERMAL_COLORMAPS = {
    0: cv2.COLORMAP_WINTER,
    1: cv2.COLORMAP_INFERNO,
    2: cv2.COLORMAP_TURBO,
    3: cv2.COLORMAP_PLASMA,
    4: cv2.COLORMAP_HOT,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

ERROR_WORDS = [
    "ERROR", "NULL", "404", "FAIL", "?????", "NO MATCH",
    "SIGNAL LOST", "TOO HUMAN", "RETRY", "0x00", "UNCLASSIFIED",
    "CANNOT READ", "DATA LOST", "ABORT", "CORRUPT"
]

dot = Dorothy(WIDTH, HEIGHT)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def lerp(current, target, amount):
    return current + (target - current) * amount


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def landmarks_center(landmarks):
    return (
        sum(lm.x for lm in landmarks) / len(landmarks),
        sum(lm.y for lm in landmarks) / len(landmarks),
    )


def hand_scale(landmarks):
    wrist = landmarks[0]
    middle_base = landmarks[9]
    scale = math.sqrt((middle_base.x - wrist.x) ** 2 + (middle_base.y - wrist.y) ** 2)
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


class StableState:
    def __init__(self, start_state=4, required_frames=8):
        self.current = start_state
        self.candidate = start_state
        self.count = 0
        self.required_frames = required_frames

    def update(self, prediction):
        if prediction == self.candidate:
            self.count += 1
        else:
            self.candidate = prediction
            self.count = 1

        if self.count >= self.required_frames:
            self.current = self.candidate

        return self.current


class MySketch:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.detector = self.make_detector()
        self.samples_x = []
        self.samples_y = []
        self.knn = None
        self.load_training_data()

        self.state_filter = StableState()
        self.state = 4
        self.expand = 0.0
        self.expand_smooth = 0.0
        self.no_hand_frames = 0
        self.frame_count = 0
        self.last_sd_state = None
        self.last_sd_frame = -9999
        self.sd_queue = queue.Queue(maxsize=1)
        self.sd_image = None
        self.sd_lock = threading.Lock()
        self.sd_ready = False
        self.last_result = None
        self.has_audio = False

        if SD_ENABLED and get_device is not None and load_pipe is not None:
            threading.Thread(target=self.sd_worker, daemon=True).start()

        dot.start_loop(self.setup, self.draw)

    def make_detector(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Put the MediaPipe hand_landmarker.task file "
                "next to this script before running."
            )

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        return vision.HandLandmarker.create_from_options(options)

    def load_training_data(self):
        if not os.path.exists(DATA_PATH):
            return

        with open(DATA_PATH, "rb") as f:
            saved = pickle.load(f)
            self.samples_x = saved.get("X", [])
            self.samples_y = saved.get("y", [])

        self.train_classifier()

    def save_training_data(self):
        with open(DATA_PATH, "wb") as f:
            pickle.dump({"X": self.samples_x, "y": self.samples_y}, f)

    def train_classifier(self):
        if len(set(self.samples_y)) < 2 or len(self.samples_x) < 6:
            self.knn = None
            return

        k = min(3, len(self.samples_x))
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.samples_x, self.samples_y)

    def setup(self):
        try:
            kwargs = dict(
                position=0.5,
                spread=0.08,
                grain_size=120.0,
                density=6.0,
                attack=0.3,
                decay=0.4,
                n_grains=12,
                pitch=0.0,
                pitch_spread=0.0,
                sr=44100,
                buffer_size=512,
            )
            path = "sample.wav"
            output_id = (
                dot.music.start_granular_stream(path=path, **kwargs)
                if os.path.exists(path)
                else dot.music.start_granular_stream(**kwargs)
            )
            self.synth = dot.music.audio_outputs[output_id]
            self.synth.note_on(160, vel=0.45)
            self.has_audio = True
        except Exception:
            self.has_audio = False

    def sd_worker(self):
        try:
            device = get_device()
            pipe = load_pipe(device)
            self.sd_ready = True
        except Exception:
            self.sd_ready = False
            return

        while True:
            frame, state = self.sd_queue.get()
            prompt = SD_PROMPTS.get(state, SD_PROMPTS[0])
            strength = SD_STRENGTH.get(state, 0.35)

            try:
                image = pipe(
                    prompt=prompt,
                    image=frame,
                    num_inference_steps=SD_STEPS,
                    strength=strength,
                    guidance_scale=SD_GUIDANCE,
                ).images[0]
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                with self.sd_lock:
                    self.sd_image = image
            except Exception:
                pass

    def detect_hands(self):
        ret, frame = self.camera.read()
        if not ret:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        return rgb, result

    def update_interaction(self, result):
        if result is None or not result.hand_landmarks:
            self.no_hand_frames += 1
            self.expand = lerp(self.expand, 0.0, 0.04)
            if self.no_hand_frames > 14:
                self.state = self.state_filter.update(4)
            return

        self.no_hand_frames = 0
        features = extract_features(result)

        if self.knn is not None:
            prediction = int(self.knn.predict([features])[0])
            self.state = self.state_filter.update(prediction)

        if len(result.hand_landmarks) >= 2:
            c1 = landmarks_center(result.hand_landmarks[0])
            c2 = landmarks_center(result.hand_landmarks[1])
            self.expand = clamp(distance(c1, c2) / 0.65)
        else:
            self.expand = lerp(self.expand, 0.0, 0.02)

    def update_sound(self):
        if not self.has_audio:
            return

        intensity = self.expand_smooth
        self.synth.grain_size = 90 + self.state * 35 + intensity * 90
        self.synth.density = int(4 + self.state * 2 + intensity * 18)
        self.synth.pitch_spread = 0.05 + self.state * 0.16 + intensity * 0.2
        self.synth.spread = 0.1 + intensity * 0.55

    def maybe_request_sd(self, rgb):
        if not SD_ENABLED or not self.sd_ready or rgb is None:
            return

        state_changed = self.state != self.last_sd_state
        waited_long_enough = (
            self.frame_count - self.last_sd_frame >= SD_MIN_FRAMES_BETWEEN_JOBS
        )

        if not state_changed and not waited_long_enough:
            return

        small = cv2.resize(rgb, (SD_SIZE, SD_SIZE))
        job = (Image.fromarray(small), self.state)

        try:
            self.sd_queue.put_nowait(job)
            self.last_sd_state = self.state
            self.last_sd_frame = self.frame_count
        except queue.Full:
            pass

    def make_thermal_scan_layer(self, display):
        gray = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        contrast = 0.9 + self.state * 0.22 + self.expand_smooth * 0.45
        saturation = 1.0 + self.state * 0.18
        thermal_input = cv2.convertScaleAbs(gray, alpha=contrast, beta=8)
        colormap = THERMAL_COLORMAPS.get(self.state, cv2.COLORMAP_TURBO)
        thermal = cv2.applyColorMap(thermal_input, colormap)

        thermal_hsv = cv2.cvtColor(thermal, cv2.COLOR_BGR2HSV).astype(np.float32)
        thermal_hsv[:, :, 1] = np.clip(thermal_hsv[:, :, 1] * saturation, 0, 255)
        thermal = cv2.cvtColor(thermal_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

        scan_y = int((self.frame_count * (2 + self.state)) % dot.height)
        band = np.zeros_like(thermal)
        band_height = 34 + int(self.expand_smooth * 70)
        y1 = max(0, scan_y - band_height)
        y2 = min(dot.height, scan_y + band_height)
        band[y1:y2, :, :] = STATE_COLORS.get(self.state, (255, 255, 255))
        band = cv2.GaussianBlur(band, (1, 61), 0)

        thermal = cv2.addWeighted(thermal, 0.82, band, 0.18, 0)
        return thermal

    def draw_camera_background(self, rgb):
        if rgb is None:
            dot.background((8, 10, 14))
            return

        display = cv2.resize(rgb, (dot.width, dot.height))
        camera_base = (display * 0.18).astype(np.uint8)
        thermal = self.make_thermal_scan_layer(display)
        base = cv2.addWeighted(camera_base, 0.48, thermal, 0.52, 0)

        with self.sd_lock:
            sd = None if self.sd_image is None else self.sd_image.copy()

        if sd is not None:
            sd = cv2.resize(sd, (dot.width, dot.height))
            sd = cv2.cvtColor(sd, cv2.COLOR_BGR2RGB)
            sd_alpha = [0.0, 0.15, 0.35, 0.65, 0.88][self.state]
            base = cv2.addWeighted(base, 1.0 - sd_alpha, sd, sd_alpha, 0)

        dot.paste(base, (0, 0))

    def draw_diagnostic_grid(self):
        color = STATE_COLORS.get(self.state, (255, 255, 255))
        grid_size = 80 - self.state * 10

        dot.fill((*color, 15 + self.state * 8))

        for x in range(0, dot.width, grid_size):
            dot.line((x, 0), (x, dot.height))

        for y in range(0, dot.height, grid_size):
            dot.line((0, y), (dot.width, y))

        coords = [
            (20, 30, "0,0"),
            (dot.width - 60, 30, f"{dot.width},0"),
            (20, dot.height - 20, f"0,{dot.height}"),
            (dot.width - 100, dot.height - 20, f"{dot.width},{dot.height}"),
        ]
        dot.fill((*color, 100))
        for x, y, label in coords:
            dot.text(label, x, y, 10)

    def draw_dynamic_core(self):
        color = STATE_COLORS.get(self.state, (255, 255, 255))
        cx, cy = dot.width // 2, dot.height // 2
        pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.08)
        intensity = self.expand_smooth

        if self.state == 0:
            width = 180 + intensity * 160
            height = 42 + pulse * 18
            y = cy - height / 2
            dot.fill((*color, 55))
            dot.rectangle((cx - width / 2, y), (width, height))
            dot.fill(color)
            scan_x = cx - width / 2 + (self.frame_count % 90) / 90 * width
            dot.line((scan_x, y - 18), (scan_x, y + height + 18))
            for offset in [-70, -35, 0, 35, 70]:
                dot.line((cx + offset, cy - 18), (cx + offset, cy + 18))

        elif self.state == 1:
            arms = 18
            base_radius = 65 + intensity * 210
            for i in range(arms):
                angle = (math.tau * i / arms) + self.frame_count * 0.012
                wobble = 18 * math.sin(self.frame_count * 0.05 + i)
                r1 = 24 + intensity * 25
                r2 = base_radius + wobble
                p1 = (cx + math.cos(angle) * r1, cy + math.sin(angle) * r1)
                p2 = (cx + math.cos(angle) * r2, cy + math.sin(angle) * r2)
                dot.fill((*color, 130))
                dot.line(p1, p2)
            dot.fill((*color, 85))
            dot.circle((cx, cy), 30 + intensity * 30)

        elif self.state == 2:
            size = 190 + intensity * 220
            step = 34
            dot.fill((*color, 65))
            dot.rectangle((cx - size / 2, cy - size / 2), (size, size))
            dot.fill(color)
            dot.line((cx - size / 2, cy), (cx + size / 2, cy))
            dot.line((cx, cy - size / 2), (cx, cy + size / 2))
            for i in range(int(size // step) + 1):
                x = cx - size / 2 + i * step
                y = cy - size / 2 + i * step
                dot.line((x, cy - 6), (x, cy + 6))
                dot.line((cx - 6, y), (cx + 6, y))
            dot.circle((cx, cy), 10 + pulse * 8)

        elif self.state == 3:
            points = []
            count = 52
            for i in range(count):
                x = cx - 300 + i * (600 / (count - 1))
                wave = math.sin(i * 0.75 + self.frame_count * 0.22)
                noise = math.sin(i * 2.1 + self.frame_count * 0.09)
                y = cy + wave * (35 + intensity * 90) + noise * 18
                points.append((x, y))
            dot.fill(color)
            for i in range(len(points) - 1):
                dot.line(points[i], points[i + 1])
            for i in range(0, len(points), 6):
                dot.rectangle((points[i][0] - 3, points[i][1] - 18), (6, 36))

        else:
            # A：故障文字雨
            for _ in range(6 + int(intensity * 8)):
                ex = random.randint(50, dot.width - 150)
                ey = random.randint(80, dot.height - 120)
                dot.fill((255, random.randint(0, 80), random.randint(0, 80)))
                dot.text(random.choice(ERROR_WORDS), ex, ey, random.randint(12, 28))

        if self.has_audio:
            synth_density = self.synth.density if hasattr(self.synth, 'density') else 0
            beat_intensity = (synth_density / 20.0) * 0.7
            dot.fill((*color, int(100 * beat_intensity)))
            dot.circle((cx, cy), 15 + beat_intensity * 120)

    def draw_hands(self, result):
        if result is None or not result.hand_landmarks:
            return

        color = STATE_COLORS.get(self.state, (255, 255, 255))
        points_list = []

        for landmarks in result.hand_landmarks:
            points = [
                (int(lm.x * dot.width), int(lm.y * dot.height))
                for lm in landmarks
            ]
            points_list.append(points)

            # E：破碎的手部骨架（ERROR 状态）
            if self.state == 4:
                dot.fill((*color, 60))
                for start, end in HAND_CONNECTIONS:
                    dot.line(points[start], points[end])

                dot.fill((255, random.randint(0, 100), random.randint(0, 100)))
                shuffled_connections = random.sample(HAND_CONNECTIONS, len(HAND_CONNECTIONS) // 2)
                for start, end in shuffled_connections:
                    jitter_s = (
                        points[start][0] + random.randint(-15, 15),
                        points[start][1] + random.randint(-15, 15)
                    )
                    jitter_e = (
                        points[end][0] + random.randint(-15, 15),
                        points[end][1] + random.randint(-15, 15)
                    )
                    dot.line(jitter_s, jitter_e)

                dot.fill((255, 100, 100))
                for idx in [0, 4, 8, 12, 16, 20]:
                    ox, oy = random.randint(-8, 8), random.randint(-8, 8)
                    dot.circle((points[idx][0] + ox, points[idx][1] + oy), random.randint(4, 10))

            else:
                dot.fill((*color, 130))
                for start, end in HAND_CONNECTIONS:
                    dot.line(points[start], points[end])
                dot.fill((*color, 235))
                for start, end in HAND_CONNECTIONS:
                    dot.line(points[start], points[end])

                for idx in [0, 4, 8, 12, 16, 20]:
                    dot.circle(points[idx], 6 + self.expand_smooth * 4)

        if len(points_list) == 2:
            p1 = points_list[0][9]
            p2 = points_list[1][9]
            dot.fill((*color, 150))
            dot.line(p1, p2)

            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            dist_label = f"DISTANCE: {self.expand_smooth:.2f}"
            dot.fill(color)
            dot.text(dist_label, mid[0] - 70, mid[1] - 15, 12)

    def draw_interface(self):
        color = STATE_COLORS.get(self.state, (255, 255, 255))

        dot.fill((255, 255, 255))
        dot.text(STATE_TEXT.get(self.state, "SYSTEM READING"), 44, dot.height - 96, 30)

        dot.fill(color)
        dot.text(f"{self.state} / {STATE_NAMES.get(self.state)}", 44, 52, 22)

        dot.fill((170, 170, 170))
        dot.text(f"EXPAND {self.expand_smooth:.2f}", 44, 78, 14)
        dot.text(f"TRAINING SAMPLES {len(self.samples_x)}", 44, 96, 14)

        if self.knn is None:
            dot.fill((255, 210, 90))
            dot.text("NO TRAINED CLASSIFIER LOADED", 44, 124, 14)

        if SD_ENABLED and not self.sd_ready:
            dot.fill((180, 180, 180))
            dot.text("SD TEXTURE OFFLINE", 44, dot.height - 42, 13)

    def draw(self):
        self.frame_count += 1

        rgb, result = self.detect_hands()
        self.last_result = result
        self.update_interaction(result)
        self.expand_smooth = lerp(self.expand_smooth, self.expand, 0.12)
        self.update_sound()
        self.maybe_request_sd(rgb)

        self.draw_camera_background(rgb)
        self.draw_diagnostic_grid()
        self.draw_hands(result)
        self.draw_dynamic_core()
        self.draw_interface()


MySketch()
