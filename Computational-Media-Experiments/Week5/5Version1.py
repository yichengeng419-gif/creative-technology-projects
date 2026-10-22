from dorothy import Dorothy
import random
import math

class Mysketch:
    def __init__(self):
        self.dot = Dorothy()
        self.time = 0
        self.dot.start_loop(self.setup, self.draw)
    def setup(self):
        file_path = "/Users/xiaogeng/Desktop/CCI/Final-project../Week5/835082__bangcorrupt__wonky-beat.wav"
        self.dot.music.start_file_stream(file_path, fft_size=500)

        self.show_beat = 0
        self.lines =[]

    def draw(self):
        dot = self.dot
        dot.background(dot.yellow)
        #Method: control the rotation speed of the spiral
        self.time += 0.06

        #method:amplitude visualization-horizontal telescopic lines
        amp = dot.music.amplitude()
        amp_width = int(amp * dot.width)

        dot.stroke(dot.blue)
        dot.line((0, dot.height //2), (amp_width, dot.height // 2))

        #Method: FFT(frequency)-Vertical energy bar
        spectrum = dot.music.fft()
        #The width of the spectrum strios
        bar_width = 6
        #Nummer of bars to draw
        freq_bins_to_use = 80

        for i, val in enumerate(spectrum[:freq_bins_to_use]):
            height = int(val * 300)
            x = i * bar_width
            y = dot.height - height

            dot.stroke(dot.purple)
            dot.line((x, dot.height), (x,y))

       # Meethod: Beat Tracking
        if dot.music.is_beat():
            self.show_beat = 5
            y = random.randint(20, dot.height - 20)
            self.lines.append([0, y, dot.width, y])
        if self.show_beat > 0:
            dot.background(dot.yellow)
            self.show_beat -= 1

        #Horizontal line diffusion effect
        dot.stroke(dot.red)
        for line in self. lines:
            x1, y, x2, _ = line
            dot.line((x1, y), (x2, y))
            line[0] -=2
            line[2] +=2

        #Method: spiral based on Trigonometry + FFT
        dot.stroke(dot.blue)
        center = (dot.width // 2, dot.height // 2)
        #Spiral length/density
        num_points = 200
        angle_speed = 0.5
        prev_x, prev_y = center
       
        for i in range(num_points):
            radius = (i * 2) + spectrum[i % freq_bins_to_use] * 100
            angle = i * 0.2 + self.time * angle_speed

            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)

            dot.line((prev_x, prev_y), (x, y))
            prev_x, prev_y = x, y

Mysketch()






