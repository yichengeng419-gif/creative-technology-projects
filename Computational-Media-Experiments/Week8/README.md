Week8-Noise, Loss and Resolution
Course overview：
  This week's lesson discussed resolution, quantization (bit depth), noise, interpolation, and lossy compression.The course teaches that when continuous signals are sampled, compressed, or interpolated, certain characteristic structures emerge, such as pixelation, color banding, random noise, or blurring. These "losses" are both technical limitations and a form of visual language that can be utilized.

Inspiration
  I hope to transform these seemingly technical concepts into visual expressions, and build a digital ribbon system consisting of noise, interpolation, and color channels.
  Sound is defined within the course as "a signal that is random and unpredictable, yet exhibits controllable behaviour".
  I transformed sin-based pseudo-noise into ribbon distortion and jitter, endowing the image with the characteristics of "signal noise".

Coding process:
Step1
  'bg_r = int(100 + math.sin(self.time * 0.6) * 60)'
  Employing the concept of interpolation from the course, I controlled the background colour using a three-channel sine function, causing it to pulsate rhythmically over time.
Step2
 'self.pos[0] += (dot.mouse_x - self.pos[0]) * self.t'
  Implementing smooth following using LERP:
Step3
  Drawing a segmented path in polar coordinates to create a "ribbon" effect.
  I deliberately reduced the graphic's resolution to render the geometry as a discrete, digital structure.

  ‘glitch = math.sin(self.time * 3 + t * 5 + layer) * 15‘
   This renders the ribbon as fluctuations, distortions and a sense of energy in the digital signal.
Step4
 ‘for j in range(-self.PIXEL_SIZE, self.PIXEL_SIZE):’
  Simulating hand-drawn, broad brushstrokes using pixel blocks

Debugging：
  1.Colour errors when using integers
  By understanding that `fill()` requires three-channel input, I resolved the "int is not subscriptable" issue.
  2.Noise amplitude too strong leading to flicker
  I reduced the noise intensity to maintain the system's visual stability.
  3.glitch = math.sin(...)
    twist = math.sin(...)
    Because I added multiple noise items to the code, the combined noise amplitude caused the curve to break, the point to jump and the ribbon shape to become completely distorted, resulting in excessive animation jitter and an unattractive appearance.

  4.By progressively reducing noise and pixel thickness, I achieved a more controlled granularity in the ribbon.


Final8.1:
After completing the ribbon design, I still felt the overall visual effect was too cluttered, so I made some adjustments.
Process：
   I abandoned the original line loop and instead initialised 1500 independent particles, each possessing distinct angle, radius, and velocity modifiers.
   Employing `math.sin` to periodically alter the particle's radius
   A high-frequency sine function (glitch_t) is applied to the particle's angle to simulate high-frequency random noise.
Error：
   AttributeError: [rect/ellipse/point] not found
  This issue forced me to revert to the most fundamental, universal plotting function `dot.line()`, and by adjusting the `stroke_weight`

Conclusion
  By adjusting num_points and PIXEL_SIZE, I have observed that:
  Resolution is a crucial element in shaping visual style.

  Whether it's the change in background color, the smooth following of position, or the change in ribbon color over time, they are all based on interpolation.

  Controlling the amplitude of noise is paramount. During initial attempts, excessive noise caused complete graphical breakdown, while ‘excessive jitter’ rendered the entire visual effect particularly chaotic. By progressively reducing noise intensity, I learned to manage the loss, transforming it into a rhythmic, controlled visual enhancement that elevates the aesthetic quality of the image rather than creating random disorder.
AI support
  I utilised an AI tool to identify and resolve specific Python errors. For instance, the ‘int is not subscriptable’ issue that occurs when passing a single value to the fill() function. This tool helped me understand the principle that colour values in Dorothy must be provided in RGB tuple format, guiding me towards correcting the data structure. This deepened my understanding of colour quantisation and multi-channel representation.
  