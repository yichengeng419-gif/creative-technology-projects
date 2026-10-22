import os
import math
import time
import pickle
import queue
import random
import runpy
import threading
import traceback
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from dorothy import Dorothy
try:
    from dorothy.DorothyWindow import DorothyWindow
except Exception:
    DorothyWindow = None
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

try:
    from diffusion_utils import get_device, load_pipe
except ImportError:
    get_device = None
    load_pipe = None


# Please Gesture More Clearly
# Performance version:
# webcam -MediaPipe hand tracking -KNN gesture classification
# thermal body scan + SD hallucination + particles + fixed diagnostic language + sound.

WIDTH, HEIGHT = 880, 495
MODEL_PATH = "hand_landmarker.task"
DATA_PATH = "gesture_data.pkl"

PERFORMANCE_DURATION = 115
TENSION_PEAK_TIME = 98
RESET_NO_HAND_FRAMES = 90

SD_SIZE = 128
SD_ENABLED = True
SD_MIN_FRAMES_BETWEEN_JOBS = 190
SD_STEPS = 2
SD_GUIDANCE = 0.0

CONFUSION_THRESHOLD = 2.0
PARTICLE_POOL = 42
DATA_POOL = 42
DATA_CHARS = "0123456789.:#%*+=ABCDEF?"
ASCII_CHARS = " .:-=+*#%@0123456789"
ASCII_CELL_W = 14
ASCII_CELL_H = 16
ASCII_REFRESH_FRAMES = 5
USE_LIVE_RULES = False
LIVE_RULES_PATH = "live_rules.py"
LIVE_RELOAD_FRAMES = 12

PHASES = [
    (0, 18, "AWAKENING"),
    (18, 46, "READING"),
    (46, 80, "OVER-READING"),
    (80, 100, "COLLAPSE"),
    (100, 115, "STILL READING"),
]

PHASE_MAX_STATE = {
    "STANDBY": 0,
    "AWAKENING": 0,
    "READING": 2,
    "OVER-READING": 3,
    "COLLAPSE": 4,
    "STILL READING": 4,
}

STATE_NAMES = {
    0: "STABLE",
    1: "EXPAND",
    2: "INTERPRETED",
    3: "OVERINTERPRETED",
    4: "BODY CANNOT BE CLASSIFIED",
}

STATE_COLORS = {
    0: (90, 220, 180),
    1: (255, 180, 80),
    2: (90, 190, 255),
    3: (220, 90, 255),
    4: (255, 70, 70),
}

DIAGNOSTIC_TEXTS = {
    0: [
        "CALLING BODY",
        "WAITING FOR HAND RESPONSE",
        "LINE OPENING",
        "CALIBRATING BODY CONNECTION",
    ],
    1: [
        "EXPANSION DETECTED",
        "BODY BECOMING RHYTHM",
        "SIGNAL AMPLIFIED",
        "GESTURE OPENING TOO WIDELY",
    ],
    2: [
        "PATTERN FOUND",
        "MACHINE CONFIDENCE RISING",
        "HAND MEANING EXTRACTED",
        "EMOTIONAL SHAPE CONFIRMED",
    ],
    3: [
        "PLEASE GESTURE MORE CLEARLY",
        "OVERINTERPRETATION IN PROGRESS",
        "HAND NOISE BECOMES EVIDENCE",
        "CONFIDENCE EXCEEDS BODY",
    ],
    4: [
        "BODY CANNOT BE CLASSIFIED",
        "CALL TO BODY FAILED",
        "NO ANSWER, STILL READING",
        "OPERATOR ERROR: TOO HUMAN",
        "LINE OPEN, BODY ABSENT",
    ],
}

ERROR_WORDS = [
    "ERROR", "NULL", "404", "FAIL", "NO MATCH", "TOO HUMAN",
    "SIGNAL LOST", "RETRY", "0x00", "UNCLASSIFIED", "CANNOT READ",
    "DATA LOST", "CORRUPT", "STILL READING", "PLEASE REPEAT BODY",
]

SD_PROMPTS = {
    0: (
        "thermal camera image of human hands as a faint incoming call signal, "
        "dark medical monitor, soft scan lines, waiting connection, biometric interface"
    ),
    1: (
        "orange infrared hand heat expanding into a body signal field, "
        "call connection opening, thermal bloom, camera texture, diagnostic light"
    ),
    2: (
        "cyan machine vision hand scan translated into data, "
        "body signal analysis, spectral skin map, clean biometric overlay, connection established"
    ),
    3: (
        "purple corrupted hand signal, overinterpreted biometric scan, "
        "fractured call data, unstable machine vision, broken thermal feedback"
    ),
    4: (
        "red emergency thermal scan, failed call to body, no answer signal, "
        "operator error screen, broken biometric connection, dense warning light"
    ),
}

SD_STRENGTH = {0: 0.55, 1: 0.62, 2: 0.68, 3: 0.76, 4: 0.84}

BGM_SOUND_FILE = "sounds/441311__aptbr__glitch-scape-beat-thing.wav"

SOUND_FILES = {
    0: "sounds/796163__anonio82__phone_call.wav",
    1: "sounds/583393__erokia__ambient-wave-marker-26_2.wav",
    2: "sounds/624393__erokia__elementary-wave-2.wav",
    3: "sounds/583393__erokia__ambient-wave-marker-26_2.wav",
    4: "sounds/619080__mrrap4food__operator-error-tone.mp3",
}

THERMAL_COLORMAPS = {
    0: cv2.COLORMAP_WINTER,
    1: cv2.COLORMAP_INFERNO,
    2: cv2.COLORMAP_TURBO,
    3: cv2.COLORMAP_PLASMA,
    4: cv2.COLORMAP_HOT,
}

PARTICLE_PARAMS = {
    0: {"count": 14, "spread": 75, "drift": 0.3},
    1: {"count": 28, "spread": 170, "drift": 0.7},
    2: {"count": 26, "spread": 130, "drift": 0.5},
    3: {"count": 42, "spread": 175, "drift": 1.2},
    4: {"count": 62, "spread": 230, "drift": 1.9},
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


if DorothyWindow is not None:
    DorothyWindow.fullscreen = False
    DorothyWindow.resizable = False


def safe_dorothy_exit(self):
    # Fullscreen startup/early shutdown can call exit before Audio is attached.
    music = getattr(self, "music", None)
    if music is not None:
        music.clean_up()


Dorothy.exit = safe_dorothy_exit

dot = Dorothy(WIDTH, HEIGHT)
CPU_CANVAS_BGR = None


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def lerp(current, target, amount):
    return current + (target - current) * amount


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def paste_bgr(frame_bgr):
    frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
    try:
        dot.paste(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (0, 0))
    except TypeError:
        dot.canvas = frame_bgr


def set_cpu_canvas_bgr(frame_bgr):
    global CPU_CANVAS_BGR
    CPU_CANVAS_BGR = np.ascontiguousarray(frame_bgr.astype(np.uint8))
    paste_bgr(CPU_CANVAS_BGR)


def get_cpu_canvas_bgr():
    if CPU_CANVAS_BGR is None:
        return np.zeros((dot.height, dot.width, 3), dtype=np.uint8)
    if CPU_CANVAS_BGR.shape[:2] != (dot.height, dot.width):
        return cv2.resize(CPU_CANVAS_BGR, (dot.width, dot.height))
    return CPU_CANVAS_BGR.copy()


def draw_text(text, x, y, size=18, color=(255, 255, 255), alpha=1.0):
    font_scale = max(0.32, size / 28)
    thickness = max(1, int(size / 14))
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    target = get_cpu_canvas_bgr()

    if alpha < 0.99:
        overlay = target.copy()
        cv2.putText(
            overlay,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            bgr,
            thickness,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, target, 1.0 - alpha, 0, target)
    else:
        cv2.putText(
            target,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            bgr,
            thickness,
            cv2.LINE_AA,
        )

    set_cpu_canvas_bgr(target)


def safe_fill(color):
    dot.fill((int(color[0]), int(color[1]), int(color[2])))


def draw_line(pt1, pt2, width=1, color=(180, 230, 255)):
    dot.stroke((int(color[0]), int(color[1]), int(color[2])))
    dot.set_stroke_weight(max(1, int(width)))
    dot.line(pt1, pt2)


def cv2_color(color):
    return tuple(int(c) for c in color[:3])


def blend_overlay_rgb(overlay_rgb, alpha=0.65):
    overlay = np.ascontiguousarray(overlay_rgb.astype(np.uint8))
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    base = get_cpu_canvas_bgr()
    if overlay_bgr.shape != base.shape:
        overlay_bgr = cv2.resize(overlay_bgr, (base.shape[1], base.shape[0]))
    mixed = cv2.addWeighted(overlay_bgr, alpha, base, 1.0, 0)
    set_cpu_canvas_bgr(mixed)


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


# Convert MediaPipe hand landmarks into a fixed-length vector for the KNN classifier.
# The coordinates are normalised around each wrist so the model reads gesture shape,
# rather than the absolute position of the hand in the camera frame.
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


# Small stabiliser for live classification.
# A new gesture state must appear for several frames before it becomes the official state,
# which prevents rapid flickering when the hand is between two learned gestures.
class StableState:
    def __init__(self, start_state=4, required_frames=7):
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
        self.current_features = None
        self.training_message = "PRESS 0-4 TO RECORD GESTURE SAMPLES"
        self.training_message_until = 0.0
        self.load_training_data()

        self.state_filter = StableState(start_state=0, required_frames=4)
        self.state = 0
        self.prev_state = 0
        self.raw_prediction = 0
        self.prev_raw_prediction = 0
        self.visual_load = 0.0
        self.expand = 0.0
        self.expand_smooth = 0.0
        self.no_hand_frames = 0
        self.frame_count = 0

        self.performance_start = None
        self.elapsed = 0.0
        self.tension = 0.0
        self.phase_name = "STANDBY"
        self.live_rules = {}
        self.live_rules_mtime = None
        self.last_live_reload_frame = -9999

        self.prev_palm = None
        self.hand_speed = 0.0
        self.hand_height = 0.5

        self.flash = 0.0
        self.burst = 0.0
        self.diagnostic_text = "SYSTEM WAITING FOR BODY"
        self.next_text_time = 0.0
        self.scan_log = deque(maxlen=5)
        self.last_logged_state = None
        self.just_changed_state = False
        self.body_echoes = deque(maxlen=7)
        self.last_echo_frame = -9999
        self.ghost_hands = []
        self.ghost_state = 0
        self.ghost_started_frame = 0
        self.last_hand_frame = -9999

        self.last_sd_state = None
        self.last_sd_frame = -9999
        self.sd_queue = queue.Queue(maxsize=1)
        self.sd_image = None
        self.sd_lock = threading.Lock()
        self.sd_ready = False
        self.layer_errors = set()
        self.ascii_overlay = None
        self.last_ascii_frame = -9999

        self.has_audio = False
        self.audio_outputs = {}
        self.audio_output = None
        self.bgm_output = None
        self.active_audio_state = None
        self.audio_gains = {}
        self.particles = []
        for _ in range(PARTICLE_POOL):
            self.particles.append({
                "angle": random.uniform(0, math.tau),
                "radius": random.uniform(0.25, 1.0),
                "speed": random.uniform(0.006, 0.035) * random.choice([-1, 1]),
                "size": random.uniform(2.0, 6.0),
                "phase": random.uniform(0, math.tau),
            })
        self.data_chars = []
        for _ in range(DATA_POOL):
            self.data_chars.append({
                "angle": random.uniform(0, math.tau),
                "radius": random.uniform(0.25, 1.0),
                "speed": random.uniform(0.004, 0.024) * random.choice([-1, 1]),
                "char": random.choice(DATA_CHARS),
                "size": random.randint(12, 24),
                "phase": random.uniform(0, math.tau),
            })

        if SD_ENABLED and get_device is not None and load_pipe is not None:
            threading.Thread(target=self.sd_worker, daemon=True).start()

        dot.on_key_pressed = self.on_key_pressed
        dot.start_loop(self.setup, self.draw)

    def draw_layer(self, name, fn, *args):
        try:
            fn(*args)
        except Exception as exc:
            if name not in self.layer_errors:
                self.layer_errors.add(name)
                print(f"DRAW LAYER FAILED [{name}]: {exc}")
                traceback.print_exc()

    # Create the MediaPipe hand detector used by the whole performance.
    # The model returns 21 landmarks per detected hand.
    def make_detector(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Put hand_landmarker.task next to this script."
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

    # Load the recorded gesture examples from gesture_data.pkl.
    # These samples are what make the performance respond to my trained hand states.
    def load_training_data(self):
        if not os.path.exists(DATA_PATH):
            return

        with open(DATA_PATH, "rb") as f:
            saved = pickle.load(f)
            self.samples_x = saved.get("X", [])
            self.samples_y = saved.get("y", [])

        self.train_classifier()

    # Train a simple KNN classifier from the recorded gesture examples.
    # KNN is useful here because it is transparent: unfamiliar gestures can be treated as confusion.
    def train_classifier(self):
        if len(set(self.samples_y)) < 2 or len(self.samples_x) < 6:
            self.knn = None
            return

        k = min(3, len(self.samples_x))
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.samples_x, self.samples_y)

    def save_training_data(self):
        with open(DATA_PATH, "wb") as f:
            pickle.dump({"X": self.samples_x, "y": self.samples_y}, f)

    def training_counts(self):
        return {state: self.samples_y.count(state) for state in range(5)}

    def set_training_message(self, text):
        self.training_message = text
        self.training_message_until = time.time() + 2.5
        print(text)

    # Keyboard recording interface: pressing 0-4 stores the current hand pose as a labelled example.
    # This keeps the system trainable during rehearsal without opening a separate GUI.
    def on_key_pressed(self, key):
        key = str(key).lower()
        if key in {"0", "1", "2", "3", "4"}:
            if self.current_features is None:
                self.set_training_message("NO HAND DETECTED - SHOW HANDS BEFORE RECORDING")
                return
            label = int(key)
            self.samples_x.append(self.current_features.copy())
            self.samples_y.append(label)
            self.train_classifier()
            self.save_training_data()
            count = self.samples_y.count(label)
            self.set_training_message(f"RECORDED STATE {label} SAMPLE #{count}")
        elif key == "s":
            self.save_training_data()
            self.train_classifier()
            counts = self.training_counts()
            self.set_training_message(f"SAVED GESTURES {counts}")
        elif key == "x":
            self.samples_x = []
            self.samples_y = []
            self.knn = None
            self.save_training_data()
            self.set_training_message("CLEARED ALL GESTURE TRAINING DATA")

    # Load audio streams once at startup.
    # The gain of each stream is later faded in/out according to the current machine-reading state.
    def setup(self):
        try:
            if os.path.exists(BGM_SOUND_FILE):
                self.bgm_output = dot.music.start_file_stream(
                    BGM_SOUND_FILE,
                    fft_size=512,
                    buffer_size=1024,
                    sr=44100,
                    analyse=True,
                )
                dot.music.audio_outputs[self.bgm_output].gain = 0.10
                self.has_audio = True

            loaded_states = []
            for state, path in SOUND_FILES.items():
                if not os.path.exists(path):
                    continue
                output = dot.music.start_file_stream(
                    path,
                    fft_size=512,
                    buffer_size=1024,
                    sr=44100,
                    analyse=True,
                )
                dot.music.audio_outputs[output].gain = 0.0
                self.audio_outputs[state] = output
                self.audio_gains[state] = 0.0
                loaded_states.append(state)

            if loaded_states or self.bgm_output is not None:
                self.has_audio = True
                print(f"BGM AUDIO LOADED: {self.bgm_output is not None}")
                print(f"STATE AUDIO LOADED: {loaded_states}")
                return

            path = "sample.wav"
            if os.path.exists(path):
                self.audio_output = dot.music.start_file_stream(
                    path,
                    fft_size=512,
                    buffer_size=1024,
                    sr=44100,
                    analyse=True,
                )
                self.has_audio = True
            else:
                self.has_audio = False
        except Exception as exc:
            print(f"AUDIO FAILED: {exc}")
            self.has_audio = False

    # Background Stable Diffusion worker.
    # SD is slow compared with the draw loop, so frames are sent through a queue and generated asynchronously.
    def sd_worker(self):
        try:
            device = get_device()
            pipe = load_pipe(device)
            self.sd_ready = True
        except Exception as exc:
            print(f"SD FAILED: {exc}")
            self.sd_ready = False
            return

        while True:
            task = self.sd_queue.get()
            if len(task) == 4:
                frame, state, tension, prompt_suffix = task
            else:
                frame, state, tension = task
                prompt_suffix = ""
            prompt = SD_PROMPTS.get(state, SD_PROMPTS[0])
            if prompt_suffix:
                prompt = f"{prompt}, {prompt_suffix}"
            steps = max(1, int(SD_STEPS))
            strength = clamp(SD_STRENGTH.get(state, 0.35) + tension * 0.08, 0.25, 0.75)
            strength = max(strength, 1.0 / steps)

            try:
                image = pipe(
                    prompt=prompt,
                    image=frame,
                    num_inference_steps=steps,
                    strength=strength,
                    guidance_scale=SD_GUIDANCE,
                ).images[0]
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                with self.sd_lock:
                    self.sd_image = image
            except Exception as exc:
                print(f"SD ERROR: {exc}")

    def detect_hands(self):
        ret, frame = self.camera.read()
        if not ret:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return rgb, result

    def reload_live_rules(self):
        if not USE_LIVE_RULES:
            self.live_rules = {}
            return

        if self.frame_count - self.last_live_reload_frame < LIVE_RELOAD_FRAMES:
            return
        self.last_live_reload_frame = self.frame_count

        if not os.path.exists(LIVE_RULES_PATH):
            self.live_rules = {}
            self.live_rules_mtime = None
            return

        try:
            mtime = os.path.getmtime(LIVE_RULES_PATH)
            if self.live_rules_mtime == mtime:
                return

            loaded = runpy.run_path(LIVE_RULES_PATH)
            self.live_rules = {
                key: value
                for key, value in loaded.items()
                if key.isupper()
            }
            self.live_rules_mtime = mtime
            print(f"LIVE RULES RELOADED: {LIVE_RULES_PATH}")
        except Exception as exc:
            print(f"LIVE RULES ERROR: {exc}")

    def rule(self, name, default=None):
        return self.live_rules.get(name, default)

    def live_scene(self):
        scene = self.rule("SCENE", "AUTO")
        if scene in {None, "AUTO", ""}:
            return self.phase_name
        return scene

    # Performance dramaturgy: once hands appear, the piece moves through timed phases.
    # This creates a one-to-two-minute arc from scanning, to reading, to over-reading, collapse, and still reading.
    def update_arc(self, result):
        has_hands = result is not None and result.hand_landmarks

        if has_hands:
            self.no_hand_frames = 0
            if self.performance_start is None:
                self.performance_start = time.time()
                self.scan_log.clear()
                self.last_logged_state = None
                self.diagnostic_text = "CALIBRATING BODY"
        else:
            self.no_hand_frames += 1
            if self.no_hand_frames > RESET_NO_HAND_FRAMES and self.elapsed < 80:
                self.performance_start = None

        if self.performance_start is None:
            self.elapsed = 0.0
            self.tension = 0.0
            self.visual_load = lerp(self.visual_load, 0.0, 0.08)
            self.phase_name = "STANDBY"
            return

        self.elapsed = time.time() - self.performance_start
        if self.elapsed < TENSION_PEAK_TIME:
            self.tension = clamp(self.elapsed / TENSION_PEAK_TIME)
        else:
            t = (self.elapsed - TENSION_PEAK_TIME) / max(1, PERFORMANCE_DURATION - TENSION_PEAK_TIME)
            self.tension = clamp(1.0 - t * 0.55)
        self.visual_load = lerp(self.visual_load, self.tension, 0.035)

        self.phase_name = "STILL READING"
        for start, end, name in PHASES:
            if start <= self.elapsed < end:
                self.phase_name = name
                break

    # Track continuous hand qualities that are not classification labels.
    # Speed, height, and spread are used to drive intensity, prompt changes, particles, and sound.
    def update_hand_dynamics(self, result):
        if result is None or not result.hand_landmarks:
            self.hand_speed = lerp(self.hand_speed, 0.0, 0.12)
            self.prev_palm = None
            return

        palm = result.hand_landmarks[0][9]
        cur = (palm.x, palm.y)
        if self.prev_palm is not None:
            raw_speed = distance(cur, self.prev_palm)
            self.hand_speed = lerp(self.hand_speed, raw_speed, 0.3)
        self.prev_palm = cur
        self.hand_height = lerp(self.hand_height, palm.y, 0.2)

    # Main interaction mapping: hand landmarks -> KNN prediction -> machine-reading state.
    # If the current gesture is far from the training set, the system treats it as confusion / cannot classify.
    def update_interaction(self, result):
        if result is None or not result.hand_landmarks:
            self.expand = lerp(self.expand, 0.0, 0.04)
            if self.no_hand_frames > 12 and self.phase_name in {"COLLAPSE", "STILL READING"}:
                self.state = self.state_filter.update(4)
            return

        features = extract_features(result)
        if self.knn is not None:
            prediction = int(self.knn.predict([features])[0])
            distances, _ = self.knn.kneighbors([features])
            nearest_dist = distances[0][0]
            effective_threshold = (
                CONFUSION_THRESHOLD
                - self.tension * 0.5
                + self.rule("CONFUSION_OFFSET", 0.0)
            )

            if nearest_dist > effective_threshold:
                prediction = 4

            fast_motion = clamp(self.hand_speed * 12)
            if self.live_scene() in {"COLLAPSE", "STILL READING"} and fast_motion > 0.45:
                prediction = 4
            elif self.phase_name not in {"COLLAPSE", "STILL READING"} and prediction == 4:
                prediction = PHASE_MAX_STATE.get(self.phase_name, 0)

            force_state = self.rule("FORCE_STATE", None)
            if force_state is not None:
                prediction = int(force_state)

            prediction = min(prediction, PHASE_MAX_STATE.get(self.phase_name, 4))
            self.raw_prediction = prediction
            self.state = self.state_filter.update(prediction)
        elif self.no_hand_frames > 12 and self.phase_name in {"COLLAPSE", "STILL READING"}:
            self.raw_prediction = 4
            self.state = self.state_filter.update(4)

        if len(result.hand_landmarks) >= 2:
            c1 = landmarks_center(result.hand_landmarks[0])
            c2 = landmarks_center(result.hand_landmarks[1])
            self.expand = clamp(distance(c1, c2) / 0.65)
        else:
            self.expand = lerp(self.expand, 0.0, 0.02)

    # Trigger short events when the classified state changes.
    # These flashes, bursts, and log entries make state changes readable in the live performance.
    def update_events(self):
        now = time.time()

        if self.state != self.prev_state:
            self.just_changed_state = True
            self.flash = 1.0
            self.burst = 1.0
            self.set_diagnostic_text(force=True)
            if self.performance_start is not None:
                self.scan_log.append((self.elapsed, STATE_NAMES.get(self.state, "?")))
            self.prev_state = self.state
        else:
            self.just_changed_state = False

        if self.raw_prediction != self.prev_raw_prediction:
            self.flash = max(self.flash, 0.42)
            self.burst = max(self.burst, 0.34)
            self.prev_raw_prediction = self.raw_prediction

        if now >= self.next_text_time:
            self.set_diagnostic_text()
            self.next_text_time = now + random.uniform(2.0, 3.8)

        if self.live_scene() == "STILL READING" and self.no_hand_frames > 8:
            self.diagnostic_text = "NO ANSWER · LINE STILL OPEN"

        self.flash = lerp(self.flash, 0.0, 0.14)
        self.burst = lerp(self.burst, 0.0, 0.08)

    # Choose machine diagnostic text for the current phase/state.
    # The language is intentionally clinical and absurd to show machine overinterpretation.
    def set_diagnostic_text(self, force=False):
        force_text = self.rule("FORCE_TEXT", None)
        if force_text:
            self.diagnostic_text = str(force_text)
            return

        if self.performance_start is None and not force:
            self.diagnostic_text = "SYSTEM WAITING FOR BODY"
            return

        choices = DIAGNOSTIC_TEXTS.get(self.state, DIAGNOSTIC_TEXTS[4])
        scene = self.live_scene()
        if scene == "AWAKENING":
            choices = [
                "CALLING BODY",
                "LINE OPENING",
                "WAITING FOR HAND RESPONSE",
                "BODY ENTERS SCAN",
            ]
        elif scene == "READING":
            choices = [
                "PATTERN FOUND",
                "GESTURE BECOMES DATA",
                "MACHINE CONFIDENCE RISING",
            ]
        elif scene == "OVER-READING":
            choices = [
                "HAND NOISE BECOMES EVIDENCE",
                "INTERPRETATION EXCEEDS SIGNAL",
                "PLEASE GESTURE MORE CLEARLY",
            ]
        elif scene == "STILL READING":
            choices = [
                "NO ANSWER, STILL READING",
                "PREDICTION CONTINUES",
                "BODY ABSENT, LINE STILL OPEN",
                "PLEASE RETURN TO BE MISREAD",
                "CALL TO BODY REMAINS ACTIVE",
            ]
        elif scene == "COLLAPSE":
            choices = DIAGNOSTIC_TEXTS[4] + DIAGNOSTIC_TEXTS[3]

        self.diagnostic_text = random.choice(choices)

    # Sound mapping.
    # BGM and per-state audio files are smoothly faded according to gesture state, tension, and hand expansion.
    def update_sound(self):
        if not self.has_audio:
            return

        if self.rule("MUTE_AUDIO", False):
            if self.bgm_output is not None:
                dot.music.pause(self.bgm_output)
            for output in self.audio_outputs.values():
                dot.music.pause(output)
            if self.audio_output is not None:
                dot.music.pause(self.audio_output)
            return

        if self.bgm_output is not None:
            bgm_gain = clamp(0.055 + self.tension * 0.050 + self.burst * 0.020, 0.045, 0.125)
            if self.phase_name == "COLLAPSE":
                bgm_gain *= 0.62
            elif self.phase_name == "STILL READING":
                bgm_gain *= 0.34
            dot.music.audio_outputs[self.bgm_output].gain = bgm_gain
            dot.music.resume(self.bgm_output)

        if self.audio_outputs:
            available = sorted(self.audio_outputs.keys())
            target_state = self.state if self.state in self.audio_outputs else min(
                available,
                key=lambda candidate: abs(candidate - self.state),
            )
            intensity = clamp(0.32 + self.expand_smooth * 0.15 + self.tension * 0.18 + self.burst * 0.14)
            if target_state == 0:
                # The phone-call sound is an opening cue, not a loop for the whole performance.
                if self.performance_start is None:
                    opening_gate = 1.0
                else:
                    opening_gate = 1.0 - clamp((self.elapsed - 10.0) / 10.0)
                intensity = clamp((0.42 + self.burst * 0.08) * opening_gate, 0.0, 0.52)
            elif target_state == 1:
                intensity = clamp(0.14 + self.expand_smooth * 0.05 + self.tension * 0.05, 0.10, 0.22)
            elif target_state == 4:
                intensity = clamp(0.22 + self.tension * 0.12 + self.burst * 0.10, 0.18, 0.42)
            elif target_state == 2:
                intensity = clamp(0.24 + self.expand_smooth * 0.08 + self.tension * 0.08 + self.burst * 0.05, 0.20, 0.38)
            elif target_state == 3:
                intensity = clamp(0.18 + self.tension * 0.10 + self.burst * 0.08, 0.14, 0.32)

            for state, output in self.audio_outputs.items():
                target_gain = intensity if state == target_state else 0.0
                current_gain = self.audio_gains.get(state, 0.0)
                fade_speed = 0.060 if target_gain > current_gain else 0.045
                new_gain = lerp(current_gain, target_gain, fade_speed)
                if new_gain < 0.004:
                    new_gain = 0.0
                self.audio_gains[state] = new_gain
                dot.music.audio_outputs[output].gain = new_gain
                if new_gain > 0.0:
                    dot.music.resume(output)
                else:
                    dot.music.pause(output)

            self.active_audio_state = target_state
            return

        if self.audio_output is not None:
            dot.music.audio_outputs[self.audio_output].gain = clamp(0.55 + self.tension * 0.25)
            dot.music.resume(self.audio_output)

    # Send occasional camera frames to Stable Diffusion.
    # This is not done every frame, because real-time diffusion is expensive; state changes and time gaps trigger updates.
    def maybe_request_sd(self, rgb):
        if not SD_ENABLED or not self.sd_ready or rgb is None:
            return

        state_changed = self.state != self.last_sd_state
        waited = self.frame_count - self.last_sd_frame >= SD_MIN_FRAMES_BETWEEN_JOBS

        if not state_changed and not waited:
            return

        small = cv2.resize(rgb, (SD_SIZE, SD_SIZE))
        prompt_suffix = self.rule("PROMPT_SUFFIX", "")
        data_words = []
        if self.expand_smooth > 0.5:
            data_words.append("expanding connection field, signal bloom")
        elif self.expand_smooth < 0.2:
            data_words.append("compressed weak signal, waiting line")
        if clamp(self.hand_speed * 10) > 0.4:
            data_words.append("motion blur, fragmented call data, signal drag")
        if self.hand_height < 0.4:
            data_words.append("rising thermal contact, luminous hand trace")
        if self.tension > 0.6:
            data_words.append("intense saturated failed connection, overwhelming machine confidence")
        if data_words:
            extra = ", ".join(data_words)
            prompt_suffix = f"{prompt_suffix}, {extra}" if prompt_suffix else extra
        try:
            self.sd_queue.put_nowait((Image.fromarray(small), self.state, self.tension, prompt_suffix))
            self.last_sd_state = self.state
            self.last_sd_frame = self.frame_count
        except queue.Full:
            pass

    # Return phase-specific visual weights.
    # Each phase changes how much SD, ASCII, particles, tearing, and text are allowed to dominate the image.
    def phase_profile(self):
        profiles = {
            "STANDBY": {
                "sd": 0.0, "grid": 0.0, "particles": 0.0,
                "errors": 0.0, "hand": 0.35, "flash": 0.0,
                "echo": 0.0, "tear": 0.0, "aura": 0.10, "data": 0.0,
            },
            "AWAKENING": {
                "sd": 0.38, "grid": 0.0, "particles": 0.05,
                "errors": 0.0, "hand": 0.45, "flash": 0.25,
                "echo": 0.02, "tear": 0.0, "aura": 0.35, "data": 0.22,
            },
            "READING": {
                "sd": 0.50, "grid": 0.0, "particles": 0.20,
                "errors": 0.0, "hand": 0.85, "flash": 0.45,
                "echo": 0.12, "tear": 0.02, "aura": 0.65, "data": 0.32,
            },
            "OVER-READING": {
                "sd": 0.68, "grid": 0.0, "particles": 0.70,
                "errors": 0.12, "hand": 1.0, "flash": 0.75,
                "echo": 0.58, "tear": 0.14, "aura": 1.0, "data": 0.48,
            },
            "COLLAPSE": {
                "sd": 0.84, "grid": 0.0, "particles": 1.25,
                "errors": 1.0, "hand": 1.0, "flash": 1.0,
                "echo": 1.05, "tear": 0.62, "aura": 1.35, "data": 1.0,
            },
            "STILL READING": {
                "sd": 0.30, "grid": 0.0, "particles": 0.05,
                "errors": 0.28, "hand": 0.25, "flash": 0.15,
                "echo": 0.48, "tear": 0.08, "aura": 0.12, "data": 0.28,
            },
        }
        scene = self.live_scene()
        profile = dict(profiles.get(scene, profiles["STANDBY"]))
        p = self.phase_progress(scene)

        if scene in {"AWAKENING", "READING", "OVER-READING"}:
            ramp = 0.25 + 0.75 * p
            for key in ["sd", "particles", "echo", "aura", "data", "tear"]:
                profile[key] *= ramp
        elif scene == "COLLAPSE":
            surge = 0.65 + 0.35 * math.sin(p * math.pi)
            for key in ["sd", "particles", "echo", "aura", "data", "tear", "errors"]:
                profile[key] *= surge
        elif scene == "STILL READING":
            fade = 1.0 - p * 0.55
            for key in ["particles", "aura", "hand"]:
                profile[key] *= fade

        arc_gate = clamp(0.12 + self.visual_load * 1.05)
        sd_gate = clamp(0.55 + self.visual_load * 0.55)
        profile["sd"] *= sd_gate
        for key in ["particles", "echo", "tear", "aura", "data", "errors", "flash"]:
            profile[key] *= arc_gate

        machine_load = self.rule("MACHINE_LOAD", 1.0)
        multipliers = {
            "sd": self.rule("SD_ALPHA", 1.0),
            "particles": self.rule("PARTICLES", 1.0),
            "echo": self.rule("BODY_ECHO", 1.0),
            "tear": self.rule("TEAR_AMOUNT", 1.0),
            "aura": self.rule("AURA", 1.0),
            "data": self.rule("DATA_TEXT", 1.0),
            "errors": self.rule("ERROR_TEXT", 1.0),
            "flash": self.rule("FLASH", 1.0),
            "hand": self.rule("HAND_SKELETON", 1.0),
        }
        for key, value in multipliers.items():
            profile[key] *= value

        for key in ["sd", "particles", "echo", "tear", "aura", "data", "errors", "flash"]:
            profile[key] *= machine_load

        return profile

    def phase_progress(self, scene=None):
        if self.performance_start is None:
            return 0.0
        scene = scene or self.phase_name
        for start, end, name in PHASES:
            if name == scene:
                return clamp((self.elapsed - start) / max(0.001, end - start))
        return 0.0

    # Store delayed snapshots of the camera image.
    # These echoes create the feeling that the machine keeps reading traces of the body after movement changes.
    def capture_body_echo(self, rgb):
        if rgb is None:
            return

        profile = self.phase_profile()
        if profile["echo"] <= 0.02:
            return

        speed = clamp(self.hand_speed * 10)
        scene = self.live_scene()
        periodic_overread = scene == "OVER-READING" and self.frame_count - self.last_echo_frame > 24
        periodic_collapse = scene == "COLLAPSE" and self.frame_count - self.last_echo_frame > 10
        should_capture = self.just_changed_state or speed > 0.45 or periodic_overread or periodic_collapse

        if not should_capture:
            return

        frame = cv2.resize(rgb, (dot.width, dot.height))
        self.body_echoes.append({
            "image": frame,
            "frame": self.frame_count,
            "state": self.state,
            "speed": speed,
        })
        self.last_echo_frame = self.frame_count

    def overlay_body_echoes(self, base):
        profile = self.phase_profile()
        echo_strength = profile["echo"]
        if echo_strength <= 0.02 or not self.body_echoes:
            return base

        output = base.copy()
        for idx, echo in enumerate(reversed(self.body_echoes)):
            age = self.frame_count - echo["frame"]
            if age > 220:
                continue

            image = echo["image"]
            state = echo["state"]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            thermal = cv2.applyColorMap(gray, THERMAL_COLORMAPS.get(state, cv2.COLORMAP_TURBO))
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

            direction = -1 if idx % 2 else 1
            drift = int((idx + 1) * 18 * echo_strength * direction)
            shake = int(math.sin(self.frame_count * 0.08 + idx) * 28 * echo_strength)
            dx = drift + shake
            dy = int(math.cos(self.frame_count * 0.05 + idx) * 10 * echo_strength)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                thermal,
                matrix,
                (dot.width, dot.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            alpha = echo_strength * max(0.0, 1.0 - age / 220) * (0.18 + idx * 0.035)
            output = cv2.addWeighted(output, 1.0 - alpha, shifted, alpha, 0)

        return output

    def tear_frame(self, image):
        amount = self.phase_profile()["tear"] + self.burst * 0.18
        if amount <= 0.02:
            return image

        torn = image.copy()
        band_height = max(8, int(34 - amount * 20))
        max_shift = int(75 * amount)

        for y in range(0, dot.height, band_height):
            wave = math.sin(y * 0.025 + self.frame_count * 0.18)
            shift = int(wave * max_shift)
            torn[y:y + band_height] = np.roll(torn[y:y + band_height], shift, axis=1)

        zoom = 1.0 + amount * 0.08
        if zoom > 1.01:
            crop_w = int(dot.width / zoom)
            crop_h = int(dot.height / zoom)
            x1 = (dot.width - crop_w) // 2
            y1 = (dot.height - crop_h) // 2
            cropped = torn[y1:y1 + crop_h, x1:x1 + crop_w]
            torn = cv2.resize(cropped, (dot.width, dot.height))

        return torn

    def glitch_signal(self, image, amount):
        # thermal -> increasingly broken / fractured signal (amount 0..1)
        if amount <= 0.02:
            return image
        out = image.copy()
        h, w = out.shape[:2]

        # 1.horizontal scan-line breaks (datamosh-like displacement)
        n_bands = int(2 + amount * 9)
        for _ in range(n_bands):
            bh = random.randint(3, max(4, int(16 * amount) + 4))
            y = random.randint(0, max(1, h - bh))
            shift = random.randint(-int(45 * amount) - 1, int(45 * amount) + 1)
            out[y:y + bh] = np.roll(out[y:y + bh], shift, axis=1)

        # 2.RGB channel split (signal desync)
        sh = int(amount * 10)
        if sh > 0:
            out[:, :, 0] = np.roll(out[:, :, 0], sh, axis=1)
            out[:, :, 2] = np.roll(out[:, :, 2], -sh, axis=1)

        # 3.dropped / corrupted blocks (signal loss), only when badly broken
        if amount > 0.4:
            n_blocks = int((amount - 0.4) * 16)
            for _ in range(n_blocks):
                bw = random.randint(8, 38)
                bh = random.randint(3, 16)
                x = random.randint(0, max(1, w - bw))
                y = random.randint(0, max(1, h - bh))
                if random.random() < 0.5:
                    out[y:y + bh, x:x + bw] = random.randint(0, 255)
                else:
                    out[y:y + bh, x:x + bw] = np.roll(out[y:y + bh, x:x + bw], random.randint(-20, 20), axis=1)

        return out

    def make_thermal_scan_layer(self, display):
        gray = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        contrast = 0.95 + self.state * 0.18 + self.expand_smooth * 0.5 + self.tension * 0.38
        saturation = 1.0 + self.state * 0.16 + self.tension * 0.35
        thermal_input = cv2.convertScaleAbs(gray, alpha=contrast, beta=8)

        colormap = THERMAL_COLORMAPS.get(self.state, cv2.COLORMAP_TURBO)
        thermal = cv2.applyColorMap(thermal_input, colormap)
        hsv = cv2.cvtColor(thermal, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        thermal = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

        scan_y = int((self.frame_count * (2 + self.state + self.tension * 3)) % dot.height)
        band = np.zeros_like(thermal)
        band_height = 26 + int(self.expand_smooth * 80 + self.tension * 40)
        y1 = max(0, scan_y - band_height)
        y2 = min(dot.height, scan_y + band_height)
        band[y1:y2, :, :] = STATE_COLORS.get(self.state, (255, 255, 255))
        band = cv2.GaussianBlur(band, (1, 65), 0)

        return cv2.addWeighted(thermal, 0.80, band, 0.20, 0)


    def ascii_amount(self):
        scene_amounts = {
            "STANDBY": 0.0,
            "AWAKENING": 0.08,
            "READING": 0.28,
            "OVER-READING": 0.48,
            "COLLAPSE": 0.64,
            "STILL READING": 0.34,
        }
        scene = self.live_scene()
        return clamp(scene_amounts.get(scene, 0.0) * (0.65 + self.visual_load * 0.55) + self.state * 0.025)

    # Translate the live camera image into characters.
    # This makes the body appear as machine-readable data instead of a natural image.
    def draw_ascii_body(self, rgb):
        amount = self.ascii_amount()
        if rgb is None or amount <= 0.025:
            return

        if self.ascii_overlay is None or self.frame_count - self.last_ascii_frame >= ASCII_REFRESH_FRAMES:
            display = cv2.resize(rgb, (dot.width, dot.height))
            gray = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 45, 115)
            signal = cv2.addWeighted(gray, 0.35, edges, 1.2, 0)

            overlay = np.zeros((dot.height, dot.width, 3), dtype=np.uint8)
            base_color = np.array(STATE_COLORS.get(self.state, (255, 255, 255)), dtype=np.float32)
            cool = np.array((135, 235, 255), dtype=np.float32)
            color = cv2_color((base_color * 0.55 + cool * 0.45).clip(0, 255))
            bgr = (color[2], color[1], color[0])

            threshold = 38 + int((1.0 - amount) * 52)
            chars = ASCII_CHARS
            font_scale = 0.34
            thickness = 1
            # Offset creates the feeling of a terminal scan moving across the body.
            scan_offset = (self.frame_count // ASCII_REFRESH_FRAMES) % 4
            for y in range(0, dot.height - ASCII_CELL_H, ASCII_CELL_H):
                for x in range(0, dot.width - ASCII_CELL_W, ASCII_CELL_W):
                    if ((x // ASCII_CELL_W) + scan_offset) % 4 == 0 and amount < 0.55:
                        continue
                    cell = signal[y:y + ASCII_CELL_H, x:x + ASCII_CELL_W]
                    value = int(cell.mean())
                    if value < threshold:
                        continue
                    idx = int(clamp(value / 255.0) * (len(chars) - 1))
                    ch = chars[idx]
                    if random.random() < amount * 0.10:
                        ch = random.choice("0123456789ABCDEF")
                    jitter = int(amount * random.uniform(-2, 2))
                    cv2.putText(
                        overlay,
                        ch,
                        (x + jitter, y + ASCII_CELL_H - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        bgr,
                        thickness,
                        cv2.LINE_AA,
                    )

            if self.state >= 3:
                # a few broken terminal rows when the reading becomes overconfident
                for _ in range(int(3 + amount * 8)):
                    y = random.randrange(0, dot.height, ASCII_CELL_H)
                    text = "".join(random.choice("01ABCDEF#%+") for _ in range(random.randint(8, 24)))
                    x = random.randint(20, max(21, dot.width - 260))
                    cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, bgr, 1, cv2.LINE_AA)

            overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
            self.ascii_overlay = overlay
            self.last_ascii_frame = self.frame_count

        alpha = clamp(0.18 + amount * 0.46)
        blend_overlay_rgb(cv2.cvtColor(self.ascii_overlay, cv2.COLOR_BGR2RGB), alpha)

    # Build the main visual background.
    # The live camera is transformed into thermal scan, blended with SD hallucination, then fractured by glitch.
    def draw_camera_background(self, rgb):
        if rgb is None:
            dot.background((8, 10, 14))
            return

        profile = self.phase_profile()
        display = cv2.resize(rgb, (dot.width, dot.height))
        camera_base = (display * (0.20 - self.tension * 0.08)).astype(np.uint8)
        thermal = self.make_thermal_scan_layer(display)
        base = cv2.addWeighted(camera_base, 0.45, thermal, 0.55, 0)

        with self.sd_lock:
            sd = None if self.sd_image is None else self.sd_image.copy()

        if sd is not None:
            sd = cv2.resize(sd, (dot.width, dot.height))
            sd = cv2.cvtColor(sd, cv2.COLOR_BGR2RGB)
            state_alpha = [0.58, 0.68, 0.76, 0.86, 0.94][self.state]
            sd_alpha = clamp(0.38 + state_alpha * profile["sd"] + self.tension * 0.18, 0.38, 0.92)
            base = cv2.addWeighted(base, 1.0 - sd_alpha, sd, sd_alpha, 0)
            base = cv2.addWeighted(base, 1.05, thermal, 0.12, 0)

        # thermal slowly fractures into broken signal as state + time rise
        glitch_amount = clamp(self.state / 4.0 * 0.48 + self.tension * 0.46)
        base = self.glitch_signal(base, glitch_amount)
        base = self.overlay_body_echoes(base)
        base = self.tear_frame(base)

        base = np.asarray(base)
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        if base.shape[2] == 4:
            base = base[:, :, :3]
        base = np.ascontiguousarray(base.astype(np.uint8))
        set_cpu_canvas_bgr(cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

    # Draw hand-responsive particles.
    # Their density and spread increase as the machine reading becomes more intense.
    def draw_particles(self, result):
        if result is None or not result.hand_landmarks:
            return

        profile = self.phase_profile()
        if profile["particles"] <= 0.02:
            return

        base_color = STATE_COLORS.get(self.state, (255, 255, 255))
        intensity = self.expand_smooth
        speed = clamp(self.hand_speed * 10)
        height_factor = 1.0 - self.hand_height
        params = PARTICLE_PARAMS.get(self.state, PARTICLE_PARAMS[0])

        count = int(params["count"] * profile["particles"] * (0.55 + intensity * 0.35 + self.tension * 0.28 + self.burst * 0.35))
        count = min(count, PARTICLE_POOL)
        spread = params["spread"] * profile["particles"] * (0.75 + intensity * 0.55 + self.tension * 0.28 + self.burst * 0.35)
        drift = params["drift"] * (1.0 + speed * 1.8 + self.tension * 0.8)

        bright = 1.0 + height_factor * 0.35 + self.flash * 0.6
        color = tuple(min(255, int(c * bright)) for c in base_color)

        for landmarks in result.hand_landmarks:
            palm = landmarks[9]
            px = int(palm.x * dot.width)
            py = int(palm.y * dot.height)

            for i in range(count):
                p = self.particles[i]
                p["angle"] += p["speed"] * (1.0 + drift)
                wobble = math.sin(self.frame_count * 0.05 + p["phase"]) * 20 * drift
                jitter = speed * random.uniform(-12, 12) + self.burst * random.uniform(-25, 25)
                r = p["radius"] * spread + wobble + jitter
                x = int(px + math.cos(p["angle"]) * r)
                y = int(py + math.sin(p["angle"]) * r - height_factor * 34)

                # glowing dots only -- no connecting/trail lines (cleaner look)
                safe_fill(color)
                dot.circle((x, y), p["size"] + intensity * 3 + self.burst * 3)

    # Draw a soft aura around the hands/body landmarks.
    # This gives the tracked body a biological signal quality without using hard geometric shapes.
    def draw_body_aura(self, result):
        if result is None or not result.hand_landmarks:
            return

        profile = self.phase_profile()
        aura = profile["aura"]
        if aura <= 0.02:
            return

        color = STATE_COLORS.get(self.state, (255, 255, 255))
        speed = clamp(self.hand_speed * 10)

        for landmarks in result.hand_landmarks:
            palm = landmarks[9]
            px = int(palm.x * dot.width)
            py = int(palm.y * dot.height)

            for idx in [4, 8, 12, 16, 20]:
                tip = landmarks[idx]
                tx = int(tip.x * dot.width)
                ty = int(tip.y * dot.height)
                vx = tx - px
                vy = ty - py
                length = max(1.0, math.sqrt(vx * vx + vy * vy))
                nx = -vy / length
                ny = vx / length

                safe_fill(color)
                for j in range(5):
                    t = (j + 1) / 6
                    phase = self.frame_count * 0.06 + idx * 0.7 + j
                    drift = math.sin(phase) * (8 + speed * 18 + aura * 10)
                    x = px + vx * t + nx * drift
                    y = py + vy * t + ny * drift
                    previous_t = max(0.0, t - 0.11 - speed * 0.04)
                    px2 = px + vx * previous_t + nx * drift * 0.45
                    py2 = py + vy * previous_t + ny * drift * 0.45
                    draw_line((px2, py2), (x, y), 1 + int(aura * 2), color)

                    if j in {2, 4}:
                        dot.circle((x, y), 2 + aura * 3 + self.expand_smooth * 2)

                if self.state >= 3:
                    safe_fill(color)
                    for j in range(3):
                        phase = self.frame_count * 0.11 + idx + j
                        tail = 25 + j * 14 + speed * 25
                        ex = tx + math.sin(phase) * tail
                        ey = ty + math.cos(phase * 1.3) * tail
                        draw_line((tx, ty), (ex, ey), 1, color)

    def draw_data_stream(self, result):
        if result is None or not result.hand_landmarks:
            return

        profile = self.phase_profile()
        amount = profile["data"]
        if amount <= 0.02:
            return

        color = STATE_COLORS.get(self.state, (255, 255, 255))
        speed = clamp(self.hand_speed * 10)
        count = min(DATA_POOL, int(DATA_POOL * amount * (0.35 + self.tension * 0.38)))
        spread = 80 + self.expand_smooth * 95 + amount * 55

        for landmarks in result.hand_landmarks:
            palm = landmarks[9]
            px = int(palm.x * dot.width)
            py = int(palm.y * dot.height)

            for i in range(count):
                d = self.data_chars[i]
                d["angle"] += d["speed"] * (1.0 + amount * 2.0 + speed)
                if random.random() < 0.035 + amount * 0.08:
                    d["char"] = random.choice(DATA_CHARS)

                wobble = math.sin(self.frame_count * 0.05 + d["phase"]) * 18 * amount
                r = d["radius"] * spread + wobble
                x = int(px + math.cos(d["angle"]) * r)
                y = int(py + math.sin(d["angle"]) * r)
                alpha = int((80 + 130 * d["radius"]) * amount)

                if self.state >= 3 and random.random() < 0.18:
                    x += random.randint(-18, 18)
                    y += random.randint(-18, 18)

                draw_text(
                    d["char"],
                    x,
                    y,
                    d["size"] + int(amount * 8),
                    color,
                    clamp(alpha, 0, 230) / 255,
                )

    def draw_hands(self, result):
        if result is None or not result.hand_landmarks:
            return

        profile = self.phase_profile()
        if profile["hand"] <= 0.02:
            return

        # Simplified + soft: glowing joint points only, no hard skeleton lines.
        overlay = np.zeros((dot.height, dot.width, 3), dtype=np.uint8)
        base_color = np.array(STATE_COLORS.get(self.state, (255, 255, 255)), dtype=np.float32)
        dot_color = cv2_color((base_color * 0.7 + 70).clip(0, 255))
        joints = [0, 4, 8, 12, 16, 20]

        for landmarks in result.hand_landmarks:
            points = [(int(lm.x * dot.width), int(lm.y * dot.height)) for lm in landmarks]
            for idx in joints:
                px, py = points[idx]
                breathe = math.sin(self.frame_count * 0.08 + idx) * 1.5
                r = 3 + self.expand_smooth * 3 + self.flash * 3 + breathe
                if self.state == 4:
                    # at ERROR the points scatter
                    px += random.randint(-5, 5)
                    py += random.randint(-5, 5)
                cv2.circle(overlay, (px, py), max(2, int(r)), dot_color, -1, cv2.LINE_AA)

        # soft glow: blur the points so they read as light, not hard dots
        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
        bright = cv2.GaussianBlur(overlay, (3, 3), 0)
        overlay = cv2.addWeighted(overlay, 0.8, bright, 0.7, 0)
        blend_overlay_rgb(overlay, 0.55 + profile["hand"] * 0.2)

    def draw_error_words(self):
        profile = self.phase_profile()
        if profile["errors"] <= 0.02:
            return

        density = int((4 + int(self.tension * 10) + (8 if self.state == 4 else 0)) * profile["errors"])
        color = STATE_COLORS.get(self.state, (255, 255, 255))

        for i in range(density):
            random.seed(self.frame_count // 8 * 100 + i)
            word = random.choice(ERROR_WORDS)
            x = random.randint(50, dot.width - 300)
            y = random.randint(80, dot.height - 120)
            size = random.randint(12, 26)
            alpha = random.randint(90, 220)
            draw_text(word, x, y, size, color, alpha / 255)
        random.seed()

    def draw_flash_pulses(self, result):
        if self.flash <= 0.02:
            return
        if result is None or not result.hand_landmarks:
            return

        profile = self.phase_profile()
        color = STATE_COLORS.get(self.state, (255, 255, 255))
        strength = self.flash * profile["flash"]
        if strength <= 0.02:
            return

        for landmarks in result.hand_landmarks:
            palm = landmarks[9]
            px = int(palm.x * dot.width)
            py = int(palm.y * dot.height)
            safe_fill(color)
            for idx in [4, 8, 12, 16, 20]:
                tip = landmarks[idx]
                tx = int(tip.x * dot.width)
                ty = int(tip.y * dot.height)
                vx = tx - px
                vy = ty - py
                length = max(1.0, math.sqrt(vx * vx + vy * vy))
                ux = vx / length
                uy = vy / length
                for i in range(3):
                    start = 0.45 + i * 0.18
                    end = start + 0.20 + strength * 0.20
                    p1 = (px + vx * start, py + vy * start)
                    p2 = (px + ux * length * end, py + uy * length * end)
                    draw_line(p1, p2, 2 + int(strength * 3), color)

    def remember_hand_ghost(self, result):
        if result is None or not result.hand_landmarks:
            return
        self.ghost_hands = [
            [(float(lm.x), float(lm.y)) for lm in landmarks]
            for landmarks in result.hand_landmarks
        ]
        self.ghost_state = self.state
        self.ghost_started_frame = self.frame_count
        self.last_hand_frame = self.frame_count

    def draw_no_hand_ghost(self):
        if not self.ghost_hands or self.performance_start is None or self.no_hand_frames < 6:
            return

        age = self.frame_count - self.last_hand_frame
        if age > 520 and self.phase_name not in {"COLLAPSE", "STILL READING"}:
            return

        state = 4 if self.phase_name in {"COLLAPSE", "STILL READING"} else self.ghost_state
        base = np.array(STATE_COLORS.get(state, STATE_COLORS[4]), dtype=np.float32)
        red = np.array((255, 55, 70), dtype=np.float32)
        cold = np.array((120, 230, 255), dtype=np.float32)
        line_color = cv2_color((base * 0.45 + red * 0.35 + cold * 0.20).clip(0, 255))
        faint_color = cv2_color((np.array(line_color) * 0.38 + 24).clip(0, 255))
        hot_color = cv2_color((np.array(line_color) * 0.55 + 255 * 0.45).clip(0, 255))

        overlay = np.zeros((dot.height, dot.width, 3), dtype=np.uint8)
        contour_order = [0, 1, 2, 3, 4, 3, 2, 5, 6, 7, 8, 7, 6, 9, 10, 11, 12,
                         11, 10, 13, 14, 15, 16, 15, 14, 17, 18, 19, 20, 19, 18, 17, 0]
        decay = clamp(1.0 - age / 520, 0.22, 1.0)
        drift_x = math.sin(self.frame_count * 0.035) * (10 + self.tension * 28)
        drift_y = -min(90, age * 0.18) + math.cos(self.frame_count * 0.028) * 8

        for hand_index, hand in enumerate(self.ghost_hands):
            pts = [np.array((hand[idx][0] * dot.width, hand[idx][1] * dot.height), dtype=np.float32) for idx in contour_order]
            palm = np.array((hand[9][0] * dot.width, hand[9][1] * dot.height), dtype=np.float32)
            wrist = np.array((hand[0][0] * dot.width, hand[0][1] * dot.height), dtype=np.float32)
            middle = np.array((hand[12][0] * dot.width, hand[12][1] * dot.height), dtype=np.float32)
            hand_size = max(40.0, float(np.linalg.norm(middle - wrist)))

            for layer in range(6):
                phase = self.frame_count * 0.052 + layer * 0.95 + hand_index
                scale = 1.0 + layer * 0.035 + self.tension * 0.04
                shifted = []
                for p in pts:
                    v = p - palm
                    q = palm + v * scale
                    q += np.array((drift_x * (layer + 1) * 0.16, drift_y - layer * 5), dtype=np.float32)
                    q += np.array((math.sin(phase + p[1] * 0.01) * (layer * 2 + 1), math.cos(phase + p[0] * 0.01) * (layer + 1)), dtype=np.float32)
                    if layer >= 3:
                        q += np.array((random.uniform(-4, 4), random.uniform(-4, 4)), dtype=np.float32)
                    shifted.append((int(q[0]), int(q[1])))
                thickness = 2 if layer < 2 else 1
                cv2.polylines(overlay, [np.array(shifted, dtype=np.int32)], True, line_color if layer < 3 else faint_color, thickness, cv2.LINE_AA)

            scan_y = int((palm[1] + drift_y + math.sin(self.frame_count * 0.08) * hand_size * 0.8) % dot.height)
            cv2.line(overlay, (0, scan_y), (dot.width, scan_y), faint_color, 1, cv2.LINE_AA)
            marker = palm + np.array((hand_size * 0.48 + drift_x, -hand_size * 0.42 + drift_y), dtype=np.float32)
            cv2.circle(overlay, (int(marker[0]), int(marker[1])), 7, hot_color, -1, cv2.LINE_AA)
            cv2.line(overlay, (int(marker[0] - 24), int(marker[1])), (int(marker[0] + 24), int(marker[1])), line_color, 1, cv2.LINE_AA)
            cv2.line(overlay, (int(marker[0]), int(marker[1] - 24)), (int(marker[0]), int(marker[1] + 24)), line_color, 1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(overlay, (19, 19), 0)
        overlay = cv2.addWeighted(glow, 0.65, overlay, 0.95, 0)
        blend_overlay_rgb(overlay, 0.38 + decay * 0.24)

        if self.no_hand_frames > 14:
            draw_text("NO ANSWER", 44, int(dot.height * 0.45), 22, line_color, 0.70)
            draw_text("LINE STILL OPEN", 44, int(dot.height * 0.45) + 30, 15, (230, 230, 230), 0.58)

    def draw_gesture_creature(self, result):
        if result is None or not result.hand_landmarks:
            self.draw_no_hand_ghost()
            return

        self.remember_hand_ghost(result)
        profile = self.phase_profile()
        speed = clamp(self.hand_speed * 10)
        state = self.state
        intensity = clamp(0.28 + profile["aura"] * 0.35 + self.expand_smooth * 0.22 + self.burst * 0.22)
        if intensity <= 0.04:
            return

        overlay = np.zeros((dot.height, dot.width, 3), dtype=np.uint8)
        base = np.array(STATE_COLORS.get(state, (255, 255, 255)), dtype=np.float32)
        cold = np.array((110, 235, 255), dtype=np.float32)
        warm = np.array((255, 178, 80), dtype=np.float32)
        violet = np.array((190, 90, 255), dtype=np.float32)
        red = np.array((255, 55, 70), dtype=np.float32)

        if state <= 0:
            line_color = cv2_color((base * 0.35 + cold * 0.65).clip(0, 255))
        elif state == 1:
            line_color = cv2_color((base * 0.42 + warm * 0.58).clip(0, 255))
        elif state == 2:
            line_color = cv2_color((base * 0.45 + cold * 0.55).clip(0, 255))
        elif state == 3:
            line_color = cv2_color((base * 0.42 + violet * 0.58).clip(0, 255))
        else:
            line_color = cv2_color((base * 0.42 + red * 0.58).clip(0, 255))
        soft_color = cv2_color((np.array(line_color) * 0.45 + 35).clip(0, 255))
        hot_color = cv2_color((np.array(line_color) * 0.55 + 255 * 0.45).clip(0, 255))

        def hp(landmarks, idx):
            lm = landmarks[idx]
            return np.array((lm.x * dot.width, lm.y * dot.height), dtype=np.float32)

        contour_order = [0, 1, 2, 3, 4, 3, 2, 5, 6, 7, 8, 7, 6, 9, 10, 11, 12,
                         11, 10, 13, 14, 15, 16, 15, 14, 17, 18, 19, 20, 19, 18, 17, 0]

        for hand_index, landmarks in enumerate(result.hand_landmarks):
            pts = [hp(landmarks, idx) for idx in contour_order]
            palm = hp(landmarks, 9)
            wrist = hp(landmarks, 0)
            middle = hp(landmarks, 12)
            hand_size = max(40.0, float(np.linalg.norm(middle - wrist)))

            # A soft heat silhouette around the real hand: a scan specimen, not an animal.
            contour = np.array([(int(p[0]), int(p[1])) for p in pts], dtype=np.int32)
            hull = cv2.convexHull(contour)
            mask = np.zeros((dot.height, dot.width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 120)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            color_layer = np.zeros_like(overlay)
            color_layer[:, :] = soft_color
            alpha = (mask.astype(np.float32) / 255.0) * (0.18 + intensity * 0.18)
            overlay[:] = (color_layer * alpha[:, :, None] + overlay * (1.0 - alpha[:, :, None])).astype(np.uint8)

            # Finger heat trails: mapped directly from body points, so the interaction reads clearly.
            for i, idx in enumerate([4, 8, 12, 16, 20]):
                tip = hp(landmarks, idx)
                base_pt = hp(landmarks, [2, 6, 10, 14, 18][i])
                direction = tip - palm
                length = max(1.0, float(np.linalg.norm(direction)))
                normal = np.array((-direction[1], direction[0]), dtype=np.float32) / length
                trail = []
                for j in range(14):
                    t = j / 13
                    wave = math.sin(t * math.pi * 2.4 + self.frame_count * 0.08 + i) * (6 + speed * 18 + state * 2)
                    p = palm * (1 - t) + tip * t + normal * wave * math.sin(t * math.pi)
                    trail.append((int(p[0]), int(p[1])))
                cv2.polylines(overlay, [np.array(trail, dtype=np.int32)], False, hot_color, 1 + int(intensity * 2), cv2.LINE_AA)

                if state >= 3:
                    end = tip + (tip - base_pt) * (0.35 + self.tension * 0.2)
                    cv2.line(overlay, (int(tip[0]), int(tip[1])), (int(end[0]), int(end[1])), line_color, 1, cv2.LINE_AA)

            # A small scan marker near the hand, not a creature.
            marker = palm + np.array((hand_size * 0.42, -hand_size * 0.38), dtype=np.float32)
            cv2.line(overlay, (int(marker[0] - 18), int(marker[1])), (int(marker[0] + 18), int(marker[1])), line_color, 1, cv2.LINE_AA)
            cv2.line(overlay, (int(marker[0]), int(marker[1] - 18)), (int(marker[0]), int(marker[1] + 18)), line_color, 1, cv2.LINE_AA)
            cv2.circle(overlay, (int(marker[0]), int(marker[1])), 5 + state, hot_color, -1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(overlay, (13, 13), 0)
        overlay = cv2.addWeighted(glow, 0.42, overlay, 0.95, 0)
        blend_overlay_rgb(overlay, 0.46 + intensity * 0.14)

    def draw_scan_log(self):
        if self.performance_start is None or not self.scan_log:
            return
        if self.phase_name in {"AWAKENING", "STILL READING"}:
            return

        x = dot.width - 320
        y0 = 54
        draw_text("SCAN LOG", x, y0, 13, (210, 210, 210), 0.65)
        for i, (elapsed, name) in enumerate(self.scan_log):
            draw_text(
                f"T+{int(elapsed):02d}s  {name}",
                x,
                y0 + 22 + i * 19,
                12,
                (220, 220, 220),
                0.6,
            )

    # Draw readable interface text for the audience: state name, diagnostic phrase, and training status.
    # This layer makes the machine's interpretation visible rather than hidden inside the code.
    def draw_interface(self):
        color = STATE_COLORS.get(self.state, (255, 255, 255))

        draw_text(self.diagnostic_text, 44, dot.height - 96, 30, (255, 255, 255))

        draw_text(STATE_NAMES.get(self.state, "UNKNOWN"), 44, 52, 22, color)

        if self.performance_start is None:
            draw_text("SHOW HANDS TO BEGIN", 44, 78, 14, (210, 210, 210), 0.65)
        else:
            scene = self.live_scene()
            label = scene if scene == self.phase_name else f"LIVE {scene}"
            draw_text(f"{label}  T+{int(self.elapsed):02d}s", 44, 78, 14, (210, 210, 210), 0.65)

        if self.knn is None:
            draw_text("NO TRAINED CLASSIFIER LOADED", 44, 106, 14, (255, 210, 90))
        elif SD_ENABLED and not self.sd_ready and self.phase_name == "STANDBY":
            draw_text("SD TEXTURE OFFLINE", 44, 106, 14, (180, 180, 180), 0.65)

        counts = self.training_counts()
        count_text = "GESTURES " + " ".join(f"{state}:{counts[state]}" for state in range(5))
        draw_text(count_text, 44, dot.height - 42, 12, (210, 210, 210), 0.58)
        if time.time() < self.training_message_until or self.knn is None:
            draw_text(self.training_message, 44, dot.height - 20, 12, (255, 230, 150), 0.72)

    # Main frame loop.
    # Each frame: read camera, classify gesture, update sound/SD requests, then draw visual layers in order.
    def draw(self):
        self.frame_count += 1

        rgb, result = self.detect_hands()
        if result is not None and result.hand_landmarks:
            self.current_features = extract_features(result)
        else:
            self.current_features = None
        self.reload_live_rules()

        self.update_arc(result)
        self.update_hand_dynamics(result)
        self.update_interaction(result)
        self.expand_smooth = lerp(self.expand_smooth, self.expand, 0.12)
        self.update_events()
        self.update_sound()
        self.maybe_request_sd(rgb)
        self.capture_body_echo(rgb)

        self.draw_layer("camera_background", self.draw_camera_background, rgb)
        self.draw_layer("ascii_body", self.draw_ascii_body, rgb)
        self.draw_layer("gesture_creature", self.draw_gesture_creature, result)
        self.draw_layer("body_aura", self.draw_body_aura, result)
        self.draw_layer("particles", self.draw_particles, result)
        self.draw_layer("data_stream", self.draw_data_stream, result)
        self.draw_layer("hands", self.draw_hands, result)
        self.draw_layer("error_words", self.draw_error_words)
        self.draw_layer("flash_pulses", self.draw_flash_pulses, result)
        self.draw_layer("scan_log", self.draw_scan_log)
        self.draw_layer("interface", self.draw_interface)


MySketch()
