from dorothy import Dorothy
import math, random
dot = Dorothy()

class Myoutput:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        dot.stroke_weight = 1
        self.t = 0.05
        self.pos = [dot.width // 2, dot.height // 2]
        self.time = 0
        self.PIXEL_SIZE = 5

        self.main_color = (255, 255, 255)     
        self.left_color = (255, 50, 50)     
        self.right_color = (50, 255, 255)     
    
    def draw(self):
        #Method:Creating a smooth oscillation using sin()
        bg_r = int(100 + math.sin(self.time * 0.6) * 60)
        bg_g = int(50 + math.sin(self.time * 0.4) * 40)
        bg_b = int(180 + math.sin(self.time * 0.9) * 80)
        dot.background((bg_r, bg_g, bg_b))
        #Method:LERP(Lineat interpolation)-Reduce tracking speed
        self.pos[0] += (dot.mouse_x - self.pos[0]) * self.t
        self.pos[1] += (dot.mouse_y - self.pos[1]) * self.t
        cx, cy = self.pos

        main_r = int(200 + math.sin(self.time * 1.0) * 40)
        main_g = int(200 + math.sin(self.time * 1.3 + 1) * 60)
        main_b = int(255 + math.sin(self.time * 1.8 + 2) * 30)
        self.main_color = (main_r, main_g, main_b)

        num_points = 120

        for layer in range(4):
            depth = layer * 30

            for i in range(num_points):
                t = i / num_points
            #Method:The sine function grows increasingly complex with each superposition.
                glitch = math.sin(self.time * 3 + t * 5 + layer) * 15
                angle = t * 16 * math.pi + self.time * 0.6 + glitch * 0.01

                base_r = 90 + depth
                r1 = base_r + math.sin(t * 7 + self.time * 0.5) * 50 + glitch
                r2 = r1 + 30
                #Method:x = cos(angle) * radius  y = sin(angle) * radius
                x1 = cx + math.cos(angle) * r1
                y1 = cy + math.sin(angle) * r1

                twist = math.sin(self.time * 4 + t * 3 + layer) * 16
                x2 = cx + math.cos(angle + 0.25) * r2 + twist
                y2 = cy + math.sin(angle + 0.25) * r2 + twist
            #Method:Manual Pixel Simulation and using a loop to produce line width
                for j in range(-self.PIXEL_SIZE, self.PIXEL_SIZE):

                    x1_j = x1 + j * math.sin(angle)
                    y1_j = y1 - j * math.cos(angle)
                    x2_j = x2 + j * math.sin(angle)
                    y2_j = y2 - j * math.cos(angle)
                #Method:By introducing a slight spatial offset (x±offset) between lines of different colours,
                    offset = 4

                    dot.stroke(self.left_color)
                    dot.line((x1_j - offset, y1_j), (x2_j - offset, y2_j))

                    dot.stroke(self.right_color)
                    dot.line((x1_j + offset, y1_j), (x2_j + offset, y2_j))

                    dot.stroke(self.main_color)
                    dot.line((x1_j, y1_j), (x2_j, y2_j))

        self.time += 0.015

Myoutput()



