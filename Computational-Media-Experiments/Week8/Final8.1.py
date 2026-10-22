from dorothy import Dorothy
import math, random
dot = Dorothy()

class Myoutput2:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        self.t = 0.08
        self.pos = [dot.width // 2, dot.height // 2]
        self.time = 0
        self.PARTICLE_SIZE = 3 #Method:Visual effects of downsampled images
        self.NUM_PARTICLES = 1500 
        # Method：Use stroke_weight to control the size of simulated points 
        dot.stroke_weight = self.PARTICLE_SIZE 

        self.particles = []
        for i in range(self.NUM_PARTICLES):
            self.particles.append({
                'angle': random.uniform(0, 2 * math.pi),
                'radius': random.uniform(20, 300),
                'speed_mod': random.uniform(0.5, 1.5)
            })
    
    def draw(self):
        bg_r = int(10 + math.sin(self.time * 0.1) * 5)
        bg_g = int(5 + math.sin(self.time * 0.15) * 5)
        bg_b = int(25 + math.sin(self.time * 0.2) * 10)
        dot.background((bg_r, bg_g, bg_b))
        
        # Method：LERP  （Smooth mouse follow）
        self.pos[0] += (dot.mouse_x - self.pos[0]) * self.t
        self.pos[1] += (dot.mouse_y - self.pos[1]) * self.t
        cx, cy = self.pos
        
        for particle in self.particles:
            #Method：Update particle movement
            radius_mod = math.sin(self.time * 2 * particle['speed_mod'] + particle['radius'] * 0.01) * 50
            radius = particle['radius'] + radius_mod
            rotation_speed = 0.01 + math.sin(self.time * 0.5) * 0.005
            particle['angle'] += rotation_speed * particle['speed_mod']
            
            #Method：Calculate position
            glitch_t = math.sin(self.time * 10 + particle['angle'] * 5) * 0.05
            final_angle = particle['angle'] + glitch_t
            
            x = cx + math.cos(final_angle) * radius
            y = cy + math.sin(final_angle) * radius
            #Method:Dynamic color based on position and time
            hue_offset = math.sin(self.time * 0.8 + radius * 0.02) * 50 
            r = int(100 + radius / 3 + math.sin(particle['angle'] + self.time * 3) * 100)
            g = int(100 + hue_offset + math.cos(particle['angle'] * 2 + self.time * 2) * 100)
            b = int(100 + radius / 2 - math.sin(particle['angle'] * 3 + self.time * 1.5) * 100)
            
            final_r = max(0, min(255, r))
            final_g = max(0, min(255, g))
            final_b = max(0, min(255, b))
            
            dot.stroke((final_r, final_g, final_b))
            #Methodd：Use dot.line to draw an extremely short line segment to simulate a point.
            dot.line((x, y), (x + 1, y)) 

        self.time += 0.015

Myoutput2()