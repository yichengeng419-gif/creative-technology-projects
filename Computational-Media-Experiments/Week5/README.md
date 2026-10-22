Week5-Audio and perception
Course overview:
This week's course focuses on learning how audio signals drive visualisation, specifically how three key aspects of audio—amplitude, spectrum, and beat—can be mapped into visual language through functions such as dot.music.amplitude(), dot.music.fft(), and dot.music.is_beat().

Through the course content, I aim to explore a visual expression that is more geometric, more abstract, and more 'perceptible as audio structure'. The process of selecting audio has sparked my interest in industrial-style music, and I wish for the visuals to embody a sense of mechanisation, regularity, and layered, progressive energy.

Inspirsation:Spiral
  The spiral has consistently been one of the most potent forms in nature, physical motion, and mathematical structures. Having studied the relationship between trigonometric functions (sin/cos) and geometric motion in my coursework, I believe employing the spiral as a visual expression can embody the cyclical fluctuations of rotational increase/expansion (angle) and radial extension (sin/cos).

  The mechanical energy of indsutrial music also mind me of:
  The robotic arm rotates
   Radar scanning
 
  Industrial-style dynamic geometric lines

Coding process：
Step1
FFT has been my most crucial visualisation tool this week.The course's explanation of the Fourier Tramsfors has taught me
  The high frequency is sharper and faster
  The low frequency is stronger and the jumping range is wider
Methods:
  FFT mapping spiral radius radius

  The value of FFT is superimposed on the spiral, so that it expands with the frequency energy.

  At the same time, keep the classic vertical bars display spectrum structure

Step2：
  The BPM od industrial loop is very high(150BPM),and each beat is very short.Iuse dot.music.is_beat as a tigger
  Trigger screen flash
  Trigger horizontal line bursting diffusion
  Enhance the rotation speed of the spiral

  The insights I uncovered during this process should not directly drive continuous visual structures using amplitude, as amplitude spikes occur too rapidly; beat tracking, however, operates on a rhythmic logic—suited for triggering 'events'. This constitutes a crucial distinction I've grasped this week.

Step3：
The core code of the spiral

prev_x, prev_y = center

for i in range(num_points):
    radius = (i * 1.8) + spectrum[i % freq_bins_to_use] * 200
    angle = (i * 0.25 + self.time * angle_speed)

    distortion = abs(math.sin(i * 0.5)) * 40  

    x = center[0] + (radius + distortion) * math.cos(angle)
    y = center[1] + (radius + distortion) * math.sin(angle)

    dot.line((prev_x, prev_y), (x, y))
    prev_x, prev_y = x, y

Key points:(Through the combination of distortion and FFT)
  How to combine mathematics(sin/cos)+high-frequency changes shape
  How to avoid line deformation caused by data range
  How to make the spiral look more "mechanical"

Dubuggling
Issues encountered during code debugging

1.The spiral only shows one line
Reason： Indentation error,calculation and drawing are not in the for loop
2.Spiral radious explosion
Reason: The FFT value is sometimes abnormally high and needs to be scaled appropriately
3.The difection of the beat line diffusion is reversed
Reason: Write as line[2]-=2 instes of +=2
4.TypeError: only length-1 arrays can be converted to Python scalars
Dorothy's visual functions (dot.line, dot.circle) only accept pure Python int numbers, but you passed in a NumPy array.
Dorothy's line/circle cannot handle this type of array → causes an error.

Conclusions
  The starting point came from the class examples of amplitude, FFT, and beat tracking
  This week I wanted to create a visual system that transforms audio energy into a dynamic, immersive graphic landscape. Instead of following the waveform or FFT literally, I explored how qualities of sound—impact, rhythm, frequency density, and motion—can become visual behaviours. My final outcome combines a fluid organic spiral with a mechanical gear-like pulse system triggered by beats. 

AI support
  During the code implementation process, I utilised AI-assisted tools to resolve technical issues. For instance, when encountering problems such as incorrect usage of `stroke_weight` in Dorothy and errors like `TypeError: “int” object is not callable`, I obtained swift explanations regarding API usage through AI. This aided my understanding of which plotting functions constitute “property assignments” rather than “method calls”.