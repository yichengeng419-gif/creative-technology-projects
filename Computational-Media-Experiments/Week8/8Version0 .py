from dorothy import Dorothy
import math

dot = Dorothy()
class Mysketch:

    def __init__(self):
        #The structure of dorthy: start : setup() + draw() dynamic loop
        dot.start_loop(self.setup, self.draw)
    def setup(self):
        #Method: Time variables used for animation drive
        self.t = 0.05
        #Method: Smooth interpolation movement
        self.pos = [0,0]
        self.time = 0
        #Method: colour interpolation
        self.a = dot.cyan
        self.b = dot.magenta
    
    #Method: Linear Interpolation(lerp)
    #Colour interpolation c = a + (b - a) * t
    def lerp_color(self, a, b, t):
        return [int(a[i] + (b[i]) * t)for i in range(3)]
    def draw(self):
        dot.background(dot.pink)
        #Method: smooth interpolation the circle toward the mouse position
        self.pos[0] = int(self.pos[0] + (dot.mouse_x - self.pos[0]) * self.t)
        self.pos[1] = int(self.pos[1] + (dot.mouse_y - self.pos[1]) * self.t)

        cx, cy = self.pos
        r=120
        
        #Method: create a simple rotating ring of points using polar coordinates
        #Method: mapping from polar to Cartesian with sin/cos
        for i in range(50):
            t = i / 50
            #Method: interpolation for animation(basic roatation using time)
            angle = t * 2 * math.pi + self.time
            #Method: convert polar coordinates to screen coordinates
            x = int(cx + math.cos(angle) * r)
            y = int(cy + math.sin(angle) * r)
            #Method: interpolate colour along the ring
            col = self.lerp_color(self.a, self.b, t)


            dot.fill(col)
            dot.circle((x, y), 6)

            #Method: central circle(original behaviour retained)
            dot.fill(dot.white)
            dot.circle((cx, cy), 20)

            #Method: increase time to animate rotation
            self.time += 0.02

Mysketch()
    