Week4-Cut and Paste Audio
Course overview
  This week's course introduces the fundamentals of digital audio processing, focusing on how physical sound is converted into digital data, and how to use this data to cut, splice, recombine, and mix sound at the sampling level.
Core creative technology: sound collage
Sampling: Extracting and reusing segments from existing audio.

Layering: Playing multiple tracks simultaneously (such as your loop and beats), adjusting their volume weights to create layering and atmosphere (e.g., achieving a chill/lo-fi style).

Collage & Alternating: Learning how to cut different audio segments and smoothly connect them using crossfade techniques to create a dynamic, non-linear audio flow, avoiding sound clutter.

Inspirsation：
The particle system in my visualizations is inspired by the gentle movement of falling snowflakes. The particles (snowflakes) create depth, dynamism, and atmosphere. The sensation snowfall evokes is one of tranquil rhythm, where order emerges from chaos. Thus, I aspire for my mix to cultivate a relaxed and soothing atmosphere, ultimately crafting a serene audiovisual experience.

Coding process：
Step1
  First, regarding audio processing and mixing, my core goal during the process was to use Python and librosa to combine three source audio files into a unified and harmonious chill mix.
  The `match_length` function is the core of sampling and synchronization. It processes the audio based on `target_len`: if the audio is too long, it is truncated (hard sampling); if it is too short, it is padded with zeros (mute), thus creating a uniform sample pool, which is the basis for all subsequent operations.

Step2:
  I want to adjust the depth and layering of each audio track in the final mix, and to layer mix and volume weights (Layering & Weighting)

  main = aligned['summer_loop'] * 0.80：The main melody loop is given the highest weight of 0.80 to ensure it dominates the mix.
  
  beat1 = aligned['beat_1'] * 0.18 / beat2 = aligned['beat_2'] * 0.12：The two rhythm tracks are assigned lower weighting, pushing them into the background to create a chill atmosphere and prevent sonic clutter.
  
  mixed = main + beat1 + beat2：All weighted audio tracks are simply added together (linear summation).

Step3:
beat1 = np.concatenate([beat1[shift1:], beat1[:shift1]])
This code moves the beat1 track forward (or backward, depending on the shift value) by a fixed amount (shift1/shift2). This is a simple collage technique that alters the starting alignment of the beats and the loop, effectively creating a "new" rhythmic pattern that avoids completely coinciding with the original beat of the main loop, thus increasing the sense of dynamic alternation in the sound.

beat2[:fade_len] *= fade_in：
This applies a linear fade-in effect at the start of track beat2, eliminating any potential abrupt 'pop' or jarring onset.
 
mixed = mixed / (max_amp * 1.2) “after adding signals, always normalise”.
Step4:
Visual part：
  The visuals are drawn line by line using a for-loop. A soft background color gradient is created from top to bottom by associating the color R/G/B values ​​with the y/H (vertical position), setting the tone for a chill atmosphere.
Reactive Particle System
  Initialize 80 particles, each with random (x, y, speed) attributes.

  In each frame draw, the particle's vertical velocity (y += sp) is amplified by audio energy * 3.

  This makes the particle system's movement speed directly respond to the music's intensity, creating a "snowfall" visual effect that emphasizes the instantaneous impact of the audio.

Debugging
1.TypeError: only length-1 arrays can be converted to Python scalars
This occurred because many of my visual parameters—particularly the y-coordinates generated using trigonometric functions and FFT-modulated values—were returned as numpy.float32 or 0-dim NumPy arrays. OpenCV, which Dorothy uses under the hood, requires primitive Python int types for all point coordinates. As a result, NumPy scalars were rejected.
2。AttributeError: “time not found in colour attributes”
Dorothy does not contain a global time property. Unrecognised attributes are internally interpreted as colour names, leading the framework to search for "time" in its colour mappings and fail.
3.cv2.error: “thickness must be an integer”
energy is a float, making the computed stroke weight a non-integer. OpenCV requires the line thickness to be an integer pixel value.
4.Certain .mp3 files loaded slower or triggered decoding warnings.
For sample-level audio editing, .wav is the most stable and compatible format. MP3 introduces unpredictable decoding behaviour due to compression codecs.

Conclusions：
The fourth week's project demonstrated how digital audio can be viewed not only as a perceptual medium, but also as a computational material that can be shaped, recombined, and transformed into visual forms. The entire process involved creating atmosphere through cutting, synchronizing, weighting, and mixing multiple audio sources.

In terms of mixing, I implemented a complete mixing workflow, including uniform length matching, volume weighting control, time offset, fade smoothing, and safe normalization. I explored how to construct a coherent, smooth, and layered musical environment from three different tracks. I found that temporal offset of the beats was particularly important; otherwise, it would become very chaotic.

Visually, the project explored how audio data affects geometry and atmosphere. Frequency energy drives the amplitude of parametric curves; soft gradations create atmosphere; snowflake-like particle systems create depth and spatial atmosphere. These elements collectively create a visual environment that matches the sonic atmosphere—calm, ethereal, layered, and constantly evolving. These visual effects are neither a literal reproduction of sound nor purely decorative, but rather a computational interpretation of how sound occupies space and time.

The debugging process further deepened my understanding of the limitations and possibilities of audio response systems. Dealing with issues such as type conversion, FFT fluctuations, inconsistent MP3 decoding, and rendering limitations made me realize that effective audiovisual design requires not only creativity but also precise control over data flow, numerical boundaries, and API behavior.
AI support
  Implementation of the wave scanline curve (utilising the function np.sin(self.t + x*0.02) combined with incremental parametric drawing via point-by-point interpolation) encountered difficulties with unresolved errors. AI assistance pinpointed specific knowledge gaps. Artificial intelligence aided in constructing loop structures, optimising syntax, and ensuring compatibility with the Dorothy plotting API.