from dorothy import Dorothy 
import librosa
import numpy as np
from util import draw_bars

dot = Dorothy(1200,300)

def mix_audio(audio):
    flipped = audio[::-1] * 0.5
    mixed = audio * 0.7 + flipped * 0.3
    return mixed.astype(np.float32)

class MySketch:
    def __init__(self):
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        audio, sr = librosa.load("Week4/834344__bangcorrupt__beat-glitch.wav")
        self.audio = mix_audio(audio)
        dot.music.start_sample_stream(self.audio, sr=sr)

    def draw(self):
        dot.background((255,255,255))
        dot.draw_waveform(dot.canvas, col=dot.black, with_playhead=True)
        draw_bars(dot.canvas, 2, 4)

MySketch()
