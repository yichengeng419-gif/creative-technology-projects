import numpy as np
import librosa
import random
from dorothy import Dorothy

AUDIO_PATHS = {
    'summer_loop': "/Users/xiaogeng/Desktop/CCI/Final-project../Week4/683506__sergequadrado__magic-loop.wav",
    'beat_1': "/Users/xiaogeng/Desktop/CCI/Final-project../Week4/814810__alexalvarez77__memento-vivere.mp3",
    'beat_2': "/Users/xiaogeng/Desktop/CCI/Final-project../Week4/817498__zhr__chill-background-music-4.wav",
}

dot = Dorothy(1200, 600)
#Method:Unify audio track length,Different audio clips have varying lengths. For mixing purposes, it need to be synchronised.
def match_length(a, target_len):
    if len(a) > target_len:
        return a[:target_len]
    elif len(a) < target_len:
        return np.concatenate([a, np.zeros(target_len - len(a), dtype=np.float32)])
    return a

def week4_chill_mix(tracks):
    min_len = min(len(t) for t in tracks.values())
    aligned = {name: match_length(a, min_len) for name, a in tracks.items()}
    main = aligned['summer_loop'] * 0.80
    beat1 = aligned['beat_1'] * 0.18
    fade_len = min_len // 4
    #Method:Use linear transitions to eliminate abrupt starts and ends.
    fade_in = np.linspace(0, 1, fade_len)
    beat2 = aligned['beat_2'] * 0.12
    beat2[:fade_len] *= fade_in
    shift1 = min_len // 6
    shift2 = min_len // 3
    #Method:Shift the audio track forwards or backwards to create new rhythmic relationships.
    beat1 = np.concatenate([beat1[shift1:], beat1[:shift1]])
    beat2 = np.concatenate([beat2[shift2:], beat2[:shift2]])
    #Method:mixed = trackA + trackB + trackC
    mixed = main + beat1 + beat2
    max_amp = np.max(np.abs(mixed))
    if max_amp > 0:
        mixed = mixed / (max_amp * 1.2)
    return mixed.astype(np.float32)

class MyOutput:
    def __init__(self):
        self.t = 0
        self.particles = [(random.randint(0,1200),
                           random.randint(0,600),
                           random.random()*0.5+0.2)
                          for _ in range(80)]
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        loaded = {}
        sr = None
        for name, path in AUDIO_PATHS.items():
            audio, current_sr = librosa.load(path, sr=None)
            loaded[name] = audio.astype(np.float32)
            if sr is None:
                sr = current_sr
        self.audio = week4_chill_mix(loaded)
        self.sr = sr
        dot.music.start_sample_stream(self.audio, sr=self.sr)

    def draw(self):
        W, H = dot.width, dot.height
        self.t += 0.03
        #Method:Use a for-loop to draw each horizontal line.Altering the colour-gradient
        for y in range(H):
            r = int(10 + (y/H)*40)
            g = int(25 + (y/H)*60)
            b = int(40 + (y/H)*70)
            dot.stroke((r,g,b))
            dot.line((0,y),(W,y))
        #Method:Each frame extracts energy from the audio as a visual parameter.
        try:
            energy = dot.music.fft_bands[0]
        except:
            energy = 0

        scan_y1 = (H/2) + np.sin(self.t)*60
        scan_y2 = (H/2) + np.sin(self.t*0.7+2)*90

        dot.stroke((120,180,220))
        dot.stroke_weight = int(1 + energy*2)
        pts1 = []
        for x in range(0,W,10):
            #Method:Parametric Curve
            y = scan_y1 + np.sin(self.t + x*0.02)*(15 + energy*20)
            pts1.append((x,int(y)))
        for i in range(len(pts1)-1):
            dot.line(pts1[i], pts1[i+1])

        dot.stroke((80,140,200))
        dot.stroke_weight = int(1 + energy*2)
        pts2 = []
        for x in range(0,W,10):
            y = scan_y2 + np.sin(self.t*0.6 + x*0.025 + 2)*(10 + energy*15)
            pts2.append((x,int(y)))
        for i in range(len(pts2)-1):
            dot.line(pts2[i], pts2[i+1])

        for i in range(len(self.particles)):
            #Method:Using multiple particles to record: (x, y, speed) Update their y-coordinate each frame and reset out-of-bounds.
            x,y,sp = self.particles[i]
            y += sp + energy*3
            if y > H:
                y = 0
                x = random.randint(0,W)
            self.particles[i] = (x,y,sp)
            dot.no_stroke()
            dot.fill((160,200,255,150))
            dot.circle((x,int(y)),3)

        dot.stroke((200,240,255))
        dot.stroke_weight = int(2 + energy*4)
        dot.draw_waveform(dot.canvas, col=(200,240,255), with_playhead=True)

MyOutput()







