import os
import math
import time
import pickle
import queue
import random
import threading
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


# Please Gesture More Clearly
# webcam -> MediaPipe hands -> KNN gesture -> ASCII body scan + SD hallucination
# + particles + diagnostic language + sound.

WIDTH, HEIGHT = 1280, 720
MODEL_PATH = "hand_landmarker.task"
DATA_PATH = "gesture_data.pkl"

PERFORMANCE_DURATION = 90
TENSION_PEAK_TIME = 76
RESET_NO_HAND_FRAMES = 90

SD_SIZE = 256
SD_ENABLED = True
SD_MIN_FRAMES_BETWEEN_JOBS = 90
SD_STEPS = 6
SD_GUIDANCE = 1.25
SD_BOOST = 1.8                # global multiplier: keeps phase progression, lifts SD overall

CONFUSION_THRESHOLD = 2.0
PARTICLE_POOL = 80

# ASCII layer
ASCII_CELL = 24              # pixel size of each character cell (smaller = denser/slower)
ASCII_EVERY = 4             # rebuild ASCII image every N frames (performance)
ASCII_CHARS = " .:-=+*#%@"  # dark -> bright

PHASES = [
    (0, 14, "AWAKENING"),
    (14, 34, "READING"),
    (34, 58, "OVER-READING"),
    (58, 76, "COLLAPSE"),
    (76, 90, "STILL READING"),
]

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
        "STABLE HAND PATTERN",
        "BODY ACCEPTED",
        "SIGNAL BELOW CONCERN",
        "CALIBRATION FEELS NORMAL",
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
        "ERROR: TOO HUMAN",
        "NO SIGNAL, STILL READING",
        "UNCLASSIFIED BODY CONTINUES",
    ],
}

ERROR_WORDS = [
    "ERROR", "NULL", "404", "FAIL", "NO MATCH", "TOO HUMAN",
    "SIGNAL LOST", "RETRY", "0x00", "UNCLASSIFIED", "CANNOT READ",
    "DATA LOST", "CORRUPT", "STILL READING", "PLEASE REPEAT BODY",
]

SD_PROMPT_VARIANTS = {
    0: [
        "calm biometric thermal scan, cold blue skin temperature map, soft medical noise",
        "quiet diagnostic scan, pale blue contour lines, sterile clinical texture",
        "gentle body scan, faint cyan grid, calm sensor readout, minimal interference",
    ],
    1: [
        "expanding orange thermal body scan, warm infrared skin glow, breathing heat map",
        "blooming amber heat signature, radiating warmth, soft sensor aura, glowing edges",
        "spreading golden thermal bloom, body warmth amplified, layered heat rings",
    ],
    2: [
        "cyan machine vision scan, hand tracking traces, transparent data texture",
        "spectral skin analysis, electric blue circuit hologram, precise medical overlay",
        "teal diagnostic mesh, body recognition artifact, scanning grid, data extraction",
    ],
    3: [
        "purple unstable thermal scan, corrupted gesture data, glitching skin map",
        "magenta fragmented sensor trails, overinterpreted machine vision, melting distortion",
        "violet broken biometric aura, hallucinated patterns, surreal psychedelic scan",
    ],
    4: [
        "red emergency body scan, unclassified human signal, broken thermal camera",
        "dense red error texture, damaged diagnostic image, warning heat map, corrupted",
        "violent crimson glitch, failed body reading, destroyed scan, chaotic interference",
    ],
}

SD_STRENGTH = {0: 0.30, 1: 0.40, 2: 0.48, 3: 0.58, 4: 0.66}

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
            features.extend([(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale])
        if side == "Left":
            left = features
        else:
            right = features
    return np.array(left + right)


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
        self.load_training_data()

        self.state_filter = StableState()
        self.state = 4
        self.prev_state = 4
        self.expand = 0.0
        self.expand_smooth = 0.0
        self.no_hand_frames = 0
        self.frame_count = 0

        self.performance_start = None
        self.elapsed = 0.0
        self.tension = 0.0
        self.phase_name = "STANDBY"

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
        self.body_echoes = deque(maxlen=4)
        self.last_echo_frame = -9999

        self.ascii_cache = None

        self.last_sd_state = None
        self.last_sd_frame = -9999
        self.sd_queue = queue.Queue(maxsize=1)
        self.sd_image = None
        self.sd_lock = threading.Lock()
        self.sd_ready = False

        self.has_audio = False
        self.particles = []
        for _ in range(PARTICLE_POOL):
            self.particles.append({
                "angle": random.uniform(0, math.tau),
                "radius": random.uniform(0.25, 1.0),
                "speed": random.uniform(0.006, 0.035) * random.choice([-1, 1]),
                "size": random.uniform(2.0, 6.0),
                "phase": random.uniform(0, math.tau),
            })

        if SD_ENABLED and get_device is not None and load_pipe is not None:
            threading.Thread(target=self.sd_worker, daemon=True).start()

        dot.start_loop(self.setup, self.draw)

    def make_detector(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Put hand_landmarker.task next to this script."
            )
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2,
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
                position=0.5, spread=0.08, grain_size=120.0, density=6.0,
                attack=0.3, decay=0.4, n_grains=12, pitch=0.0,
                pitch_spread=0.0, sr=44100, buffer_size=512,
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
        except Exception as exc:
            print(f"AUDIO FAILED: {exc}")
            self.has_audio = False

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
            frame, prompt, strength = self.sd_queue.get()
            try:
                image = pipe(
                    prompt=prompt, image=frame, num_inference_steps=SD_STEPS,
                    strength=strength, guidance_scale=SD_GUIDANCE,
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
            if self.no_hand_frames > RESET_NO_HAND_FRAMES:
                self.performance_start = None

        if self.performance_start is None:
            self.elapsed = 0.0
            self.tension = 0.0
            self.phase_name = "STANDBY"
            return

        self.elapsed = time.time() - self.performance_start
        if self.elapsed < TENSION_PEAK_TIME:
            self.tension = clamp(self.elapsed / TENSION_PEAK_TIME)
        else:
            t = (self.elapsed - TENSION_PEAK_TIME) / max(1, PERFORMANCE_DURATION - TENSION_PEAK_TIME)
            self.tension = clamp(1.0 - t * 0.55)

        self.phase_name = "STILL READING"
        for start, end, name in PHASES:
            if start <= self.elapsed < end:
                self.phase_name = name
                break

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

    def update_interaction(self, result):
        if result is None or not result.hand_landmarks:
            self.expand = lerp(self.expand, 0.0, 0.04)
            if self.no_hand_frames > 12:
                self.state = self.state_filter.update(4)
            return
        features = extract_features(result)
        if self.knn is not None:
            prediction = int(self.knn.predict([features])[0])
            distances, _ = self.knn.kneighbors([features])
            nearest_dist = distances[0][0]
            effective_threshold = CONFUSION_THRESHOLD - self.tension * 0.5
            if nearest_dist > effective_threshold:
                prediction = 4
            fast_motion = clamp(self.hand_speed * 12)
            if self.phase_name in {"COLLAPSE", "STILL READING"} and fast_motion > 0.45:
                prediction = 4
            self.state = self.state_filter.update(prediction)
        elif self.no_hand_frames > 12:
            self.state = self.state_filter.update(4)

        if len(result.hand_landmarks) >= 2:
            c1 = landmarks_center(result.hand_landmarks[0])
            c2 = landmarks_center(result.hand_landmarks[1])
            self.expand = clamp(distance(c1, c2) / 0.65)
        else:
            self.expand = lerp(self.expand, 0.0, 0.02)

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

        if now >= self.next_text_time:
            self.set_diagnostic_text()
            self.next_text_time = now + random.uniform(2.0, 3.8)

        if self.phase_name == "STILL READING" and self.no_hand_frames > 8:
            self.diagnostic_text = "NO SIGNAL · PREDICTION CONTINUES"

        self.flash = lerp(self.flash, 0.0, 0.14)
        self.burst = lerp(self.burst, 0.0, 0.08)

    def set_diagnostic_text(self, force=False):
        if self.performance_start is None and not force:
            self.diagnostic_text = "SYSTEM WAITING FOR BODY"
            return
        choices = DIAGNOSTIC_TEXTS.get(self.state, DIAGNOSTIC_TEXTS[4])
        if self.phase_name == "STILL READING":
            choices = [
                "NO SIGNAL, STILL READING",
                "PREDICTION CONTINUES",
                "BODY ABSENT, MODEL CONFIDENT",
                "PLEASE RETURN TO BE MISREAD",
            ]
        elif self.phase_name == "COLLAPSE":
            choices = DIAGNOSTIC_TEXTS[4] + DIAGNOSTIC_TEXTS[3]
        self.diagnostic_text = random.choice(choices)

    def update_sound(self):
        if not self.has_audio:
            return
        intensity = self.expand_smooth
        speed = clamp(self.hand_speed * 10)
        t = self.tension
        self.synth.grain_size = 80 + self.state * 38 + intensity * 110
        self.synth.density = int((4 + self.state * 3 + intensity * 22) * (1.0 + t * 0.8))
        self.synth.pitch_spread = (
            0.05 + self.state * 0.18 + intensity * 0.24 + speed * 0.35
        ) * (1.0 + t * 0.75)
        self.synth.spread = clamp(0.12 + intensity * 0.55 + t * 0.28, 0.0, 1.0)

    def build_sd_prompt(self):
        """Smarter SD: random variant per state + words driven by live hand data."""
        variants = SD_PROMPT_VARIANTS.get(self.state, SD_PROMPT_VARIANTS[0])
        prompt = random.choice(variants)
        extra = []
        # hand openness -> expanding / blooming
        if self.expand_smooth > 0.5:
            extra.append("expanding, blooming outward")
        elif self.expand_smooth < 0.2:
            extra.append("tight, dense, compressed")
        # hand speed -> motion artifacts
        if clamp(self.hand_speed * 10) > 0.4:
            extra.append("motion blur, fragmented streaks")
        # hand height -> rising / sinking
        if self.hand_height < 0.4:
            extra.append("rising upward, vertical light")
        elif self.hand_height > 0.7:
            extra.append("sinking, heavy downward shadow")
        # tension -> intensity
        if self.tension > 0.6:
            extra.append("intense, saturated, overwhelming")
        if extra:
            prompt = prompt + ", " + ", ".join(extra)
        return prompt

    def maybe_request_sd(self, rgb):
        if not SD_ENABLED or not self.sd_ready or rgb is None:
            return
        state_changed = self.state != self.last_sd_state
        waited = self.frame_count - self.last_sd_frame >= SD_MIN_FRAMES_BETWEEN_JOBS
        if not state_changed and not waited:
            return
        small = cv2.resize(rgb, (SD_SIZE, SD_SIZE))
        prompt = self.build_sd_prompt()
        # strength drifts a little with hand openness for continuous variation
        strength = clamp(SD_STRENGTH.get(self.state, 0.35)
                         + self.tension * 0.08
                         + self.expand_smooth * 0.06, 0.25, 0.78)
        try:
            self.sd_queue.put_nowait((Image.fromarray(small), prompt, strength))
            self.last_sd_state = self.state
            self.last_sd_frame = self.frame_count
        except queue.Full:
            pass

    def phase_profile(self):
        profiles = {
            "STANDBY": {"sd": 0.0, "particles": 0.0, "errors": 0.0, "hand": 0.35, "flash": 0.0, "echo": 0.0, "tear": 0.0},
            "AWAKENING": {"sd": 0.05, "particles": 0.10, "errors": 0.0, "hand": 0.45, "flash": 0.25, "echo": 0.05, "tear": 0.0},
            "READING": {"sd": 0.16, "particles": 0.28, "errors": 0.0, "hand": 0.85, "flash": 0.45, "echo": 0.25, "tear": 0.03},
            "OVER-READING": {"sd": 0.40, "particles": 0.65, "errors": 0.12, "hand": 1.0, "flash": 0.75, "echo": 0.55, "tear": 0.16},
            "COLLAPSE": {"sd": 0.78, "particles": 1.15, "errors": 1.0, "hand": 1.0, "flash": 1.0, "echo": 0.95, "tear": 0.55},
            "STILL READING": {"sd": 0.34, "particles": 0.08, "errors": 0.28, "hand": 0.25, "flash": 0.15, "echo": 0.40, "tear": 0.08},
        }
        return profiles.get(self.phase_name, profiles["STANDBY"])

    def make_ascii_layer(self, display):
        """Render the camera frame as coloured ASCII characters on black."""
        cols = dot.width // ASCII_CELL
        rows = dot.height // ASCII_CELL
        gray = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (cols, rows))

        canvas = np.zeros((dot.height, dot.width, 3), dtype=np.uint8)
        color = STATE_COLORS.get(self.state, (255, 255, 255))
        n = len(ASCII_CHARS)
        # brighten characters a little with tension so collapse glows
        bright = 1.4 + self.tension * 0.3
        col = tuple(min(255, int(c * bright)) for c in color)

        for j in range(rows):
            for i in range(cols):
                v = int(small[j, i])
                idx = min(int(v / 256 * n), n - 1)
                ch = ASCII_CHARS[idx]
                if ch == " ":
                    continue
                x = i * ASCII_CELL
                y = j * ASCII_CELL + ASCII_CELL - 4
                cv2.putText(canvas, ch, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1, cv2.LINE_AA)
        return canvas

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
            gray = echo["gray"]
            col = STATE_COLORS.get(echo["state"], (255, 255, 255))
            # tint grayscale with the state colour (single-colour ghost, matches ASCII)
            tinted = (gray[:, :, None].astype(np.float32) * (np.array(col) / 255.0)).astype(np.uint8)
            direction = -1 if idx % 2 else 1
            drift = int((idx + 1) * 18 * echo_strength * direction)
            shake = int(math.sin(self.frame_count * 0.08 + idx) * 28 * echo_strength)
            dx = drift + shake
            dy = int(math.cos(self.frame_count * 0.05 + idx) * 10 * echo_strength)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(tinted, matrix, (dot.width, dot.height),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
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
        return torn

    def capture_body_echo(self, rgb):
        if rgb is None:
            return
        profile = self.phase_profile()
        if profile["echo"] <= 0.02:
            return
        speed = clamp(self.hand_speed * 10)
        periodic_over = self.phase_name == "OVER-READING" and self.frame_count - self.last_echo_frame > 24
        periodic_coll = self.phase_name == "COLLAPSE" and self.frame_count - self.last_echo_frame > 10
        if not (self.just_changed_state or speed > 0.45 or periodic_over or periodic_coll):
            return
        gray = cv2.cvtColor(cv2.resize(rgb, (dot.width, dot.height)), cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        self.body_echoes.append({"gray": gray, "frame": self.frame_count, "state": self.state})
        self.last_echo_frame = self.frame_count

    def draw_camera_background(self, rgb):
        if rgb is None:
            dot.background((6, 6, 10))
            return
        profile = self.phase_profile()
        display = cv2.resize(rgb, (dot.width, dot.height))

        # ── background layer: SD + echoes ──
        base = np.zeros((dot.height, dot.width, 3), dtype=np.uint8)

        with self.sd_lock:
            sd = None if self.sd_image is None else self.sd_image.copy()
        if sd is not None:
            sd = cv2.resize(sd, (dot.width, dot.height))
            sd = cv2.cvtColor(sd, cv2.COLOR_BGR2RGB)
            # raised floor so SD is clearly visible even at STABLE
            state_alpha = [0.42, 0.52, 0.62, 0.74, 0.85][self.state]
            sd_alpha = clamp(state_alpha * profile["sd"] * SD_BOOST + self.tension * 0.10, 0.35, 0.95)
            base = cv2.addWeighted(base, 1.0 - sd_alpha, sd, sd_alpha, 0)

        base = self.overlay_body_echoes(base)
        # lighter dimming so SD stays bright (was 0.55)
        base = (base * 0.82).astype(np.uint8)
        base = self.tear_frame(base)

        # ── ASCII layer ON TOP, blended at reduced strength so it doesn't bury SD ──
        if self.frame_count % ASCII_EVERY == 0 or self.ascii_cache is None:
            self.ascii_cache = self.make_ascii_layer(display)
        base = cv2.addWeighted(base, 1.0, self.ascii_cache, 0.3, 0)

        dot.paste(base, (0, 0))

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

        count = int(params["count"] * profile["particles"] * (1.0 + intensity * 0.7 + self.tension * 0.6 + self.burst))
        count = min(count, PARTICLE_POOL)
        spread = params["spread"] * profile["particles"] * (1.0 + intensity + self.tension * 0.5 + self.burst * 0.7)
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
                alpha = int(95 + 120 * math.sin(self.frame_count * 0.05 + p["phase"]))
                alpha = max(35, min(230, alpha))
                if speed > 0.28 or self.state >= 3:
                    tx = int(x - math.cos(p["angle"]) * (speed + self.tension) * 28)
                    ty = int(y - math.sin(p["angle"]) * (speed + self.tension) * 28)
                    dot.fill((*color, int(alpha * 0.45)))
                    dot.line((x, y), (tx, ty))
                if self.state >= 2:
                    dot.fill((*color, 20 + int(self.tension * 35)))
                    dot.line((px, py), (x, y))
                dot.fill((*color, alpha))
                dot.circle((x, y), p["size"] + intensity * 3 + self.burst * 3)

    def draw_hands(self, result):
        if result is None or not result.hand_landmarks:
            return
        profile = self.phase_profile()
        if profile["hand"] <= 0.02:
            return
        color = STATE_COLORS.get(self.state, (255, 255, 255))
        point_sets = []
        for landmarks in result.hand_landmarks:
            points = [(int(lm.x * dot.width), int(lm.y * dot.height)) for lm in landmarks]
            point_sets.append(points)

            dot.fill((*color, int(90 * profile["hand"])))
            for start, end in HAND_CONNECTIONS:
                dot.line(points[start], points[end])

            dot.fill((*color, int(240 * profile["hand"])))
            for start, end in HAND_CONNECTIONS:
                if self.state == 4 and random.random() < 0.35:
                    js = (points[start][0] + random.randint(-12, 12), points[start][1] + random.randint(-12, 12))
                    je = (points[end][0] + random.randint(-12, 12), points[end][1] + random.randint(-12, 12))
                    dot.line(js, je)
                else:
                    dot.line(points[start], points[end])

            for idx in [0, 4, 8, 12, 16, 20]:
                radius = 6 + self.expand_smooth * 4 + self.flash * 5
                if self.state == 4:
                    ox, oy = random.randint(-7, 7), random.randint(-7, 7)
                    dot.circle((points[idx][0] + ox, points[idx][1] + oy), radius)
                else:
                    dot.circle(points[idx], radius)

        if len(point_sets) == 2:
            p1, p2 = point_sets[0][9], point_sets[1][9]
            dot.fill((*color, int(100 * profile["hand"])))
            dot.line(p1, p2)

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
            dot.fill((*color, alpha))
            dot.text(word, x, y, size)
        random.seed()

    def draw_flash(self):
        if self.flash <= 0.02:
            return
        profile = self.phase_profile()
        color = STATE_COLORS.get(self.state, (255, 255, 255))
        alpha = int(150 * self.flash * profile["flash"])
        dot.fill((*color, alpha))
        dot.rectangle((0, 0), (dot.width, dot.height))

    def draw_scan_log(self):
        if self.performance_start is None or not self.scan_log:
            return
        if self.phase_name in {"AWAKENING", "STILL READING"}:
            return
        x = dot.width - 320
        y0 = 54
        dot.fill((210, 210, 210, 160))
        dot.text("SCAN LOG", x, y0, 13)
        for i, (elapsed, name) in enumerate(self.scan_log):
            dot.fill((220, 220, 220, 150))
            dot.text(f"T+{int(elapsed):02d}s  {name}", x, y0 + 22 + i * 19, 12)

    def draw_interface(self):
        color = STATE_COLORS.get(self.state, (255, 255, 255))
        dot.fill((255, 255, 255))
        dot.text(self.diagnostic_text, 44, dot.height - 96, 30)
        dot.fill(color)
        dot.text(STATE_NAMES.get(self.state, "UNKNOWN"), 44, 52, 22)
        dot.fill((210, 210, 210, 150))
        if self.performance_start is None:
            dot.text("SHOW HANDS TO BEGIN", 44, 78, 14)
        else:
            dot.text(f"{self.phase_name}  T+{int(self.elapsed):02d}s", 44, 78, 14)
        if self.knn is None:
            dot.fill((255, 210, 90))
            dot.text("NO TRAINED CLASSIFIER LOADED", 44, 106, 14)
        elif SD_ENABLED and not self.sd_ready and self.phase_name == "STANDBY":
            dot.fill((180, 180, 180, 160))
            dot.text("SD TEXTURE OFFLINE", 44, 106, 14)

    def draw(self):
        self.frame_count += 1
        rgb, result = self.detect_hands()

        self.update_arc(result)
        self.update_hand_dynamics(result)
        self.update_interaction(result)
        self.expand_smooth = lerp(self.expand_smooth, self.expand, 0.12)
        self.update_events()
        self.update_sound()
        self.maybe_request_sd(rgb)
        self.capture_body_echo(rgb)

        self.draw_camera_background(rgb)
        self.draw_particles(result)
        self.draw_hands(result)
        self.draw_error_words()
        self.draw_flash()
        self.draw_scan_log()
        self.draw_interface()


MySketch()