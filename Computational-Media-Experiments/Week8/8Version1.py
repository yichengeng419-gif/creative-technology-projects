from dorothy import Dorothy
import math

dot = Dorothy()

class Mysketch2:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        dot.stroke_weight = 2
        self.t = 0.05
        self.pos = [dot.width//2, dot.height//2]
        self.time = 0

    def draw(self):
        r = int(20 + math.sin(self.time * 0.4) * 40)
        g = int(20 + math.sin(self.time * 0.6 + 2) * 30)
        b = int(40 + math.sin(self.time * 0.8 + 4) * 60)
        dot.background((r, g, b))

        self.pos[0] += (dot.mouse_x - self.pos[0]) * self.t
        self.pos[1] += (dot.mouse_y - self.pos[1]) * self.t
        cx, cy = self.pos
        
        for i in range(180):
            t = i / 180
            angle = t * 4 * math.pi + self.time

            r1 = 80 + math.sin(t * 6 + self.time * 0.8) * 40
            r2 = r1 + 30

            x1 = cx + math.cos(angle) * r1
            y1 = cy + math.sin(angle) * r1

            x2 = cx + math.cos(angle + 0.2) * r2
            y2 = cy + math.sin(angle + 0.2) * r2

            brightness = int(180 * (1 - t) + 60)
            dot.stroke((brightness, brightness, brightness))

            dot.line((x1, y1), (x2, y2))

        self.time += 0.01

Mysketch2()

