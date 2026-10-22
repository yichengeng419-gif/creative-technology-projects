*Process Code Summaries*

The versions saved in the Process folder are not the entirety of my development. The project went through a great deal of repeated debugging and trial-and-error — many changes, failed attempts and small adjustments were never saved as separate files, but were continuously overwritten as I iterated. What I have included here are the most representative milestones, chosen to show how the work evolved from an initial hand-tracking prototype into the final performance system.

*performing.py - First hand tracking test*
This is the first technical test of the project. The code uses a laptop camera and MediaPipe HandLandmarker to detect up to two hands in real time. It plots the hand skeletons directly onto the camera image and calculates a simple `hand_expand_amount` value based on the distance between the centres of the two hands.The main purpose of this version is to verify whether the body can serve as a real-time input source. The interaction is very intuitive: When the hands move apart, the expansion value increases; when they move closer together, the value decreases. 

*performing1.py*
This version introduces a machine learning classification feature. I have implemented a KNN classifier and defined five gesture states: `STABLE`, `EXPAND`, `INTERPRETED`, `OVERINTERPRETED` and `ERROR`. The code extracts 84 hand landmarks by measuring the position of each landmark relative to the wrist, and subsequently saves the annotated gesture samples to the `gesture_data.pkl` file.Performers can press the `0-4` keys to record different gestures for three seconds, after which the classifier is trained. Once training is complete, the system predicts the current gesture state in real time and displays the results on the webcam image.

The machine no longer merely ‘observes’ the hands, but begins to assign labels to bodily movements. This lays the foundation for the subsequent development of a system concept aimed at ‘over-interpreting the body’.
Its limitation lies in the fact that the output remains primarily diagnostic text and OpenCV visualisations. Whilst this demonstrates the effectiveness of the classification function, it has not yet resulted in a compelling performance effect for the audience.

*performing2.py* 
This version kept the MediaPipe and KNN gesture classifier, but added OSC output using `pythonosc`. The system sends two values to another program: `/gesture_state` and `/hand_expand`. This made the code more modular because Python could focus on hand tracking and classification, while another visual or sound system could receive the data.
Technically, this version introduced a clearer signal flow:

Camera -> MediaPipe hand landmarks -> KNN gesture state -> OSC messages -> external output system`

*performing3.py* 
This version developed the Dorothy performance further by adding more visual layers. It introduced a “data curtain” of vertical scanning lines, state-based geometric structures, stronger feedback, particles, and glitch rectangles during the `ERROR` state..

This version was useful because it tested escalation. The performance could move from a calmer state into a more chaotic machine-reading state. It made the overall structure closer to a one-to-two-minute live performance arc.
However, this version also revealed a major aesthetic problem. The state-based rectangles, circles, grids, and line structures made the work feel too geometric and abstract. The concept was about the body being converted into machine data, but the visuals sometimes looked like generic generative graphics rather than a specific bodily diagnostic system.

-What I learned:
More layers do not automatically make the performance stronger. The system needed a clearer visual language. This led to later changes such as thermal camera treatment, Stable Diffusion image layers, ASCII/body-as-data effects, sound cues, and a more focused “machine calling/reading the body” concept.

*performing4.py - First Dorothy performance version*

This version moved the project from OpenCV into Dorothy. It kept the camera, MediaPipe, KNN classifier, gesture recording, and five gesture states, but added a performance window, particles, feedback, diagnostic text, and sound control.

It also introduced phone OSC input through `pythonosc`. Phone accelerometer data could trigger a `kick`, which spawned particles and created more sudden visual events. This made the system feel more performative because the performer could control not only hand gestures but also impact-like moments.

The code also experimented with Ollama-generated diagnostic text. When the gesture state changed, the system asked a local language model to produce short uppercase machine-like messages. This connected to the course's constrained language / machine text ideas. If the model failed, fallback diagnostic phrases were used.

Sound was controlled through Dorothy's granular audio system. Gesture state affected grain size, density, pitch spread, and stereo spread. This made the hand classification influence both visual and sonic behaviour.The limitation was visual clarity. The camera image, hand rings, particles, text, and feedback were all present, but the visual identity was still not strong enough. The hand rings also felt too geometric and did not fully match the later concept.

-What I learned
This was the first version that felt like a real performance system, but it was too dependent on visible interface elements. I needed to make the output more atmospheric and conceptually specific.

*performing5.py*
This version marks the first time I’ve organized the system into a complete 90-second performance piece. The previous code was primarily focused on testing gesture classification, SD generation, and visual feedback, but this version introduces a clear temporal structure: AWAKENING / READING / OVER-READING / COLLAPSE / STILL READING. This transforms the work from merely gesture-triggered effects into a performance arc that progresses from scanning and reading to over-interpretation and ultimately the system’s collapse.

This version also incorporates an ASCII body data layer. The camera feed is converted into character images, making the body appear as if it has been machine-translated into numbers and symbols. This aligns closely with the project’s concept: the body is not truly understood, but rather transformed by the system into data that is readable, classifiable, and open to misinterpretation.

At the same time, performing5.py enhances Stable Diffusion’s presence. I increased SD’s generation parameters and blending strength, and added SD_BOOST, so that AI-generated images serve not merely as background textures but as the primary visual layer through which the machine “imagines the body.” On the audio front, this version also maps gesture states, hand distance, and tension to granular sound, making the sound denser and more unstable as the system “reads” the body.

However, the issue with this version is that the ASCII layer is too dominant. Because the visuals use ASCII as the base layer, the SD images are often buried beneath it, resulting in AI-generated visuals that aren’t prominent enough. Conceptually, “the body becoming data” holds true, but the visual impact is diminished.


*performing6.py*
performing6.py is a version of performing5.py with adjustments to the visual hierarchy. It does not overturn the structure of the previous version but addresses a key issue: the SD image was not prominent enough.
In this version, I have rearranged the order of image compositing. performing5.py first generated the ASCII base and then overlaid the SD on top; performing6.py, however, first establishes a dark background, places the SD as the primary visual layer, and only then overlays the body echoes, tear effect, and a lighter ASCII layer. This ensures that the SD is no longer obscured by the ASCII art but instead becomes the core illusory layer of the image.

This version also introduces SD_PROMPT_VARIANTS. Each state no longer has just one fixed prompt but features multiple prompt variants. The system randomly selects different visual descriptions based on the gesture state and continues to expand the prompt based on real-time body data: hand distance affects expansion/compression, hand speed affects motion blur/fragmented streaks, hand position affects rising/sinking, and high tension adds saturated/overwhelming effects. This makes the SD images richer and more like the machine continuously generating new misinterpretations in response to bodily changes.

At the same time, to make the screen less cluttered, performing6.py reduces the ASCII density by adjusting ASCII_CELL from 20 to 24, ASCII_EVERY from 3 to 4, and reducing the number of body echoes. This results in a clearer display and more stable performance.
Therefore, the improvements in `performing6.py` compared to `performing5.py` are as follows: it retains the complete performance structure while rebalancing the visual hierarchy, giving SD, ASCII, afterimages, and glitch effects their own distinct places. Ultimately, the visual output resembles a machine’s hallucination of the body rather than a camera feed overlaid with a character filter.



Overall process reflection
1.testing whether hands could be tracked reliably;
2.turning hand movement into trainable gesture classes;
3.sending gesture data through OSC as performance control data;
4.integrating gesture states into Dorothy sound and visual output;
5.adding Stable Diffusion img2img as machine-generated body hallucination;
6.translating the camera image into ASCII / data-based body imagery;
8.testing stronger feedback, diagnostic text, particles, glitch, and tearing effects

In summary, the project developed from a simple hand-tracking test into a live performance system combining machine learning, AI-generated visuals, and real-time sound. The final work is not about accurately recognising the body, but about showing how a machine over-interprets bodily data into diagnosis, image, and sound.
