import threading
import time
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny


def get_device():
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_prompts(pipe, prompts, device=None):
    """Encode a list of text prompts to embedding tensors once, ahead of time.

    Returns a list of tensors. Interpolate between them with slerp_embeds and
    pass the result via prompt_embeds= to LivePipe.push().
    """
    if device is None:
        device = get_device()
    embeds = []
    for p in prompts:
        emb, _ = pipe.encode_prompt(
            prompt=p,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        embeds.append(emb)
    return embeds


def slerp_embeds(t, a, b):
    """Spherical linear interpolation between two embedding tensors.
    
    Feels smoother than straight lerp in the model's semantic space. Falls
    back to lerp where the two vectors are nearly parallel.
    """
    a_n = a / a.norm(dim=-1, keepdim=True)
    b_n = b / b.norm(dim=-1, keepdim=True)
    dot_p = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot_p)
    sin_omega = torch.sin(omega)
    eps = 1e-6
    safe = sin_omega.abs() > eps
    lerp = (1 - t) * a + t * b
    slerp = (torch.sin((1 - t) * omega) / sin_omega) * a + \
            (torch.sin(t * omega) / sin_omega) * b
    return torch.where(safe, slerp, lerp)


def load_pipe(device=None):
    if device is None:
        device = get_device()
    # CPU must use float32 — fp16 is not supported on CPU
    if device == "cpu":
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sd-turbo"
        ).to(device)
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd"
        ).to(device)
    else:
        # mps and cuda both support float16
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=torch.float16
        ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def grab_webcam(cap, size):
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (size, size))


class LivePipe:
    """
    Persistent background diffusion worker.

    Features extracted from week5-3-continuous-canvas.py:
      - Warmup: two dummy passes on startup to prime the GPU graph compiler
      - Similarity filter: skips inference when the input hasn't changed much
        (cosine similarity on a 32x32 thumbnail), with a max_skips ceiling so
        it still refreshes occasionally
      - Temporal interpolation: while the worker generates frame N+1, the draw
        loop blends between N-1 and N based on elapsed time, hiding the latency

    Usage:
        live = LivePipe(load_pipe())
        live.warmup()                                    # in setup()
        live.push(frame, prompt=..., strength=..., ...)  # in draw(), non-blocking
        result = live.get_frame()                        # smooth interpolated output
        raw    = live.result                             # last completed output (for feedback)
    """

    def __init__(self, pipe, gen_size=256, sim_threshold=0.985, max_skips=8):
        self.pipe = pipe
        self.gen_size = gen_size
        self.sim_threshold = sim_threshold
        self.max_skips = max_skips

        self._latest = None              # (frame_rgb, kwargs) waiting for worker
        self._input_lock = threading.Lock()

        self._last_output = None         # most recent completed result (numpy RGB)
        self._prev_output = None         # one before that, for interpolation
        self._output_lock = threading.Lock()
        self._frame_started_at = 0.0
        self._frame_duration = 0.1       # rolling EMA of inference time

        self._prev_small = None          # 32x32 thumbnail of last processed input
        self._skipped = 0
        self._running = True

        threading.Thread(target=self._loop, daemon=True).start()

    def warmup(self):
        """Two dummy passes to warm up GPU graph compilation / kernel cache."""
        img = Image.new("RGB", (self.gen_size, self.gen_size))
        for _ in range(2):
            self.pipe(prompt="warmup", image=img,
                      num_inference_steps=2, strength=0.5, guidance_scale=0.0)

    def push(self, frame, **kwargs):
        """Hand the latest frame to the worker. Non-blocking."""
        frame = cv2.resize(frame, (self.gen_size, self.gen_size))
        with self._input_lock:
            self._latest = (frame, kwargs)

    def get_frame(self):
        """
        Return the temporally-interpolated display frame (numpy RGB), or None.
        While the worker generates the next result, this blends smoothly between
        the previous two completed outputs.
        """
        with self._output_lock:
            cur = self._last_output
            prev = self._prev_output
            elapsed = time.perf_counter() - self._frame_started_at
            dur = max(self._frame_duration, 1e-3)
        if cur is None:
            return None
        if prev is not None and elapsed < dur:
            t = min(1.0, elapsed / dur)
            return cv2.addWeighted(prev, 1.0 - t, cur, t, 0.0)
        return cur

    @property
    def result(self):
        """Last completed output — use this for feedback blending, not get_frame()."""
        with self._output_lock:
            return self._last_output

    @property
    def diffuse_fps(self):
        return 1.0 / max(self._frame_duration, 1e-3)

    def _cosine_sim(self, a, b):
        a = a.astype(np.float32).ravel()
        b = b.astype(np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _loop(self):
        while self._running:
            with self._input_lock:
                item = self._latest
            if item is None:
                time.sleep(0.005)
                continue

            frame, kwargs = item
            small = cv2.resize(frame, (32, 32))
            if (self._prev_small is not None
                    and self._skipped < self.max_skips
                    and self._cosine_sim(small, self._prev_small) > self.sim_threshold):
                self._skipped += 1
                time.sleep(0.005)
                continue

            self._skipped = 0
            self._prev_small = small

            # Clamp strength so int(steps * strength) >= 1, avoiding empty tensor error
            steps = kwargs.get("num_inference_steps", 2)
            kwargs["strength"] = max(kwargs.get("strength", 0.5), 1.0 / steps)

            t0 = time.perf_counter()
            result = self.pipe(image=Image.fromarray(frame), **kwargs).images[0]
            dt = time.perf_counter() - t0

            with self._output_lock:
                self._prev_output = self._last_output
                self._last_output = np.array(result)
                self._frame_started_at = time.perf_counter()
                self._frame_duration = 0.8 * self._frame_duration + 0.2 * dt
