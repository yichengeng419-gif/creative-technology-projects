import cv2
import numpy as np
from dorothy import Dorothy


dot = Dorothy()

class DynamicMosaic:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        self.camera = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.color = (100, 100, 200)
        self.alpha = 0.1
        self.default_block = 25
        self.block_range = (6, 40)

    def apply_mosaic(self, img, block_size):
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def draw(self):
        success, frame = self.camera.read()
        if not success:
            dot.background(dot.black)
            return
        frame = cv2.resize(frame, (dot.width, dot.height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        b, g, r = 200, 100, 100
        block_size = self.default_block
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cx = x + w // 2
            cy = y + h // 2
            dx = (cx - dot.width / 2) / (dot.width / 2)
            dy = (cy - dot.height / 2) / (dot.height / 2)
            warm = max(0, 1 - (abs(dx) + abs(dy)))
            r = int(100 + 155 * warm)
            g = int(80 + 40 * warm)
            b = int(255 - 200 * warm)
            pb, pg, pr = self.color
            b = int(pb * (1 - self.alpha) + b * self.alpha)
            g = int(pg * (1 - self.alpha) + g * self.alpha)
            r = int(pr * (1 - self.alpha) + r * self.alpha)
            self.color = (b, g, r)
            face_area = w * h
            closeness = min(1.0, max(0.0, face_area / (dot.width * dot.height * 0.25)))
            block_size = int(self.block_range[1] - (self.block_range[1] - self.block_range[0]) * closeness)
            block_size = max(self.block_range[0], min(self.block_range[1], block_size))
        mosaic = self.apply_mosaic(frame, block_size)
        color_layer = np.full_like(frame, self.color, dtype=np.uint8)
        blended = cv2.addWeighted(mosaic, 0.7, color_layer, 0.3, 0)
        dot.canvas = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

DynamicMosaic()
