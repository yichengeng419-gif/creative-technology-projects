Week7-Videos
Course overview:
  This week's course taught me how to use visual methods to demonstrate "how machines perceive motion," how to use camera input, and how to make programs understand changes over time and technically, the course covered how to use OpenCV's cv2.createBackgroundSubtractorMOG2() function to detect motion by comparing the current frame with a background model.

  I want to explore how to expand classroom demonstrations of background subtraction to create a more expressive, visually responsive system that can react to the movement of people in the camera, making it more interactive and engaging.

Note：I created two versions. The first one involved some experiments based on the course foundation, and I will explain my version 2.

Inspiration
One point I was deeply drawn to in the course was that computer "attention" is actually defined by difference. I hope my output can take the form of digital perception, where algorithms constantly rewrite their own memory of the world.

From an aesthetic perspective, I draw inspiration from long-exposure photography, motion trails in film, and ghostly images in surveillance footage. I believe this interactive approach is interesting as the content of this lesson, as it can overlay time using techniques such as frame blending, color mapping, and blurring.

Coding process
  Step1
  I started with simple webcam input and applied the MOG2 background subtractor. This generated a grayscale mask where moving pixels appeared white and stationary areas appeared black.
  Through threshold processing and the application of erosion and dilation operations, I can reduce noise and highlight the primary areas of motion.

  step2
   Turn on the camera. Then, prepare a variable to store the previous frame for later comparison.

  Step3
  Example:
  def draw(self):
    success, frame = self.camera.read()
    if not success:
        dot.background(dot.black)
        return

    frame = cv2.resize(frame, (dot.width, dot.height))
    fgmask = self.fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
  After setting up the camera, capture each frame from the camera and use background subtraction (fgbg.apply) to highlight the motion. Finally, thresholding converts the grayscale motion intensity to pure black/white.

  Step4
  After obtaining the binary motion mask, we applied morphological operations to refine and visualize the detected motion. The `cv2.erode()` method removes small, isolated noise points, and the `cv2.dilate()` method dilates the remaining active regions, making the motion appear more continuous. We then used the softening method `cv2.GaussianBlur()` to soften the edges, making the transition between motion and stillness smoother and more natural.

  Step5
  Following the optimisation of the motion mask, I transformed the detected activity into an expressive visual form by applying colour mapping and frame blending techniques.
  Utilising colour mapping and frame blending via cv2.applyColorMap(), the intensity of motion is translated into a spectrum of hues—cool tones signifying subtle shifts, while warmer shades denote more pronounced movement.
  This chromatic transformation enables the algorithm to articulate the vigour of motion, converting numerical variations into a compelling visual language.

Debuggling:
1.Even minute changes in lighting can trigger persistent noise. The detected mask flickers rapidly, motion becomes undetectable, and the final output is extremely unstable.
2.During the initial test, the motion trajectory accumulated too strongly, quickly filling the screen, resulting in no clear boundary between new and old frames and a blurry image.
3.Processing motion data issue-The initial motion mask was extremely unstable and noisy—even tiny fluctuations in lighting or camera grain were misinterpreted as motion. While erosion could remove noise, it also erased fine motion details; dilation often resulted in overly thick edges, appearing unnatural. After repeating the operations, I found that repetition also introduced visual flicker and caused inconsistencies in the final image between frames.
4.Visual blurring issue – Continuous frame blending (cv2.addWeighted()) causes the image to become excessively blurry and ghosted over time. Because each frame retains a portion of the previous frame, motion trajectories begin to merge, resulting in decreased visual sharpness. When color adjustments are combined with high-intensity color mapping, the image appears "muddy" and very difficult to distinguish.

Note:
  if self.prev_frame is None:
    self.prev_frame = frame.copy()
trail = cv2.addWeighted(self.prev_frame, self.trail_alpha, frame, 1 - self.trail_alpha, 0)
self.prev_frame = trail.copy()
  This code achieves temporal superimposition and continuity by performing a weighted blend (cv2.addWeighted) between the previous frame (self.prev_frame) and the current frame (frame).

  The experiment was extended using the cv2.addWeighted() consecutive frame blending function. This technique allows the system to retain traces of previous motion, producing visible afterimages and simulating the persistence of visual memory.
Reference::
https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

Conclusions
  In this course outcome, I used background subtraction algorithms as a foundation to transform machine vision into a poetic experiment on perception and memory.The system achieves temporal memory through motion detection and inter-frame blending, while expressing the intensity of perception through a cool-toned chromatic mood spectrum.
  From a learning perspective, this project deepened my understanding of how morphological filtering refines machine vision, how temporal blending simulates digital memory, and how colour theory can infuse emotional qualities into data-driven imagery.
AI support
  When encountering errors with OpenCV functions such as cv2.addWeighted() and cv2.applyColorMap(), ChatGPT assisted in identifying incorrect parameters and provided viable alternatives.
This enabled me to understand how parameter order and data types influence the outcomes of visual blending and colour mapping.