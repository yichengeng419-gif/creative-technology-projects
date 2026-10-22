from dorothy import Dorothy
import random
import math
BACKGROUND_COLOR = (20, 0, 90)
LINE_THICKNESS = 3

class week5output:
    def __init__(self):
        self.dot = Dorothy()
        self.time = 0
        self.gear_radius = None
        self.dot.start_loop(self.setup, self.draw)

    def setup(self):
        file_path = "/Users/xiaogeng/Desktop/CCI/Final-project../Week5/833081__looplicator__160-bpm-industrial-glitch-loop-15592-wav.wav"
        self.dot.music.start_file_stream(file_path, fft_size=800)

    def draw(self):
        dot = self.dot
        dot.background(BACKGROUND_COLOR)

        dot.stroke_weight = LINE_THICKNESS



        center = (dot.width // 2, dot.height // 2)
        self.time += 0.15

        amp = dot.music.amplitude()
        amp_width = int(amp * dot.width * 1.5)

        dot.stroke(dot.red)
        dot.line((0, dot.height // 2), (amp_width, dot.height // 2))

        spectrum = dot.music.fft()
        bar_width = 6
        freq_bins_to_use = 120

        dot.stroke((0, 255, 100))
        for i, val in enumerate(spectrum[:freq_bins_to_use]):
            height = int(val * 400)
            x = i * bar_width
            y = dot.height - height
            dot.line((x, dot.height), (x, y))

        if dot.music.is_beat():
            self.gear_radius = 0

        if self.gear_radius is not None:
            self.gear_radius += 12
            teeth = 12
            tooth_depth = 25

            dot.stroke(dot.black)
            prev_x = None
            prev_y = None
            for i in range(teeth * 2):
                r = self.gear_radius + (tooth_depth if i % 2 == 0 else 0)
                angle = (i / (teeth * 2)) * math.tau
                x = center[0] + r * math.cos(angle)
                y = center[1] + r * math.sin(angle)
                if prev_x is not None:
                    dot.line((prev_x, prev_y), (x, y))
                prev_x, prev_y = x, y

            end_x = center[0] + (self.gear_radius + tooth_depth) * math.cos(0)
            end_y = center[1] + (self.gear_radius + tooth_depth) * math.sin(0)
            dot.line((prev_x, prev_y), (end_x, end_y))

        dot.stroke(dot.purple)

        num_points = 300
        angle_speed = 0.8
        fluidity = 0.03

        prev_x, prev_y = center

        for i in range(num_points):
            angle = i * 0.15 + self.time * angle_speed

            radius = (
                i * 2
                + spectrum[i % freq_bins_to_use] * 900
                + math.sin(i * fluidity + self.time) * 50
                + math.cos(i * fluidity * 0.5 + self.time * 0.5) * 30
            )

            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)

            dot.line((prev_x, prev_y), (x, y))
            prev_x, prev_y = x, y

week5output()
