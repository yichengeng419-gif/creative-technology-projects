import random
from cv2 import circle

def draw_bars(canvas, bars=2, beats=4):
    h, w, _ = canvas.shape
    total_beats = bars * beats
    beat_spacing = w // total_beats
    bar_spacing = beat_spacing * beats
    particles_per_line = 50

    for i in range(bars):
        x_pos_bar = i * bar_spacing
        for _ in range(particles_per_line * 3):
            y = random.randint(0, h)
            jitter_x = random.randint(-5, 5)
            jitter_y = random.randint(-2, 2)
            col = (255, random.randint(120, 180), random.randint(120, 180))
            circle(canvas, (x_pos_bar + jitter_x, y + jitter_y), 4, col, -1)

        for j in range(1, beats):
            x_pos_beat = x_pos_bar + j * beat_spacing
            for _ in range(particles_per_line):
                y = random.randint(0, h)
                jitter_x = random.randint(-3, 3)
                jitter_y = random.randint(-1, 1)
                col = (random.randint(100,150), random.randint(50,80), 255)
                circle(canvas, (x_pos_beat + jitter_x, y + jitter_y), 3, col, -1)
