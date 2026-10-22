# Concept and Intent
This is a gesture-driven live performance in which a machine attempts to ‘interpret’ my body through hand tracking and machine learning, yet it consistently misreads and over-interprets my movements, eventually crashing—at which point it blames me displaying a prompt that reads ‘Please make your gestures clearer’. Machine learning is not merely a decorative element: the KNN classifier *is* the work itself. The work literally tells the story of a system that repeatedly fails to classify the human body, and so technology and concept are inextricably linked. I did not incorporate machine learning merely because the project required it; I adopted classification technology because classification errors are the very subject of the work.
The core idea I wish to convey is the arrogance of algorithmic systems—they presume they have the right to classify the human body, and attribute their own failures to humans.

# What the Misreading Revealed (my key observation)#
The most striking thing I noticed during performance was that the machine's misreadings were more compelling than any "correct," ordered output would have been. When the system broke down — fracturing the thermal image into glitched signal, hallucinating through Stable Diffusion, scattering its own tracking — the result felt as if the machine had a kind of agency of its own, rather than producing the tidy, legible, "reasonable" image a human would design. This was an accident that became central to the work: the failure state is where the piece feels most alive. It suggests that what we read as a machine "having a mind" is often just its breakdown — its refusal to stay orderly — and that we project intention onto error. 

# Contextualisation within Data / Algorithmic Performance
Memo Akten, Learning to See(2018). Akten’s neural network can only interpret real-time camera footage as objects it has been trained to recognise—clouds, waves, fire—so it projects familiar things onto unfamiliar places. The notion that ‘what we see is not the things as they are, but as we are’ aligns perfectly with my core mechanism: my classifier can only recognise the body as one of a few known poses, forcing everything else into these categories or into errors. Akten regards the machine’s misidentification as beautiful and inspiring; my work, however pushes it towards the absurd and the accusatory.
  
Lauren McCarthy.McCarthy’s work—such as her portrayal of a real-life ‘Alexa’, observing and interpreting people—highlights the discomfort of being observed and interpreted by a system. My work also addresses the issue of machines as interpreters, but transforms interpretation into a public failure rather than a seamless service.

Refik Anadol (as a contrast/foil).Anadol makes data sublime—smooth, beautiful,monumental. I deliberately positioned my work against this: I wanted data to look error-prone, anxious and absurd, not transcendent. Naming this contrast helped me understand my own aesthetic choices.

# Process and Iteration 
The most challenging aspect of this project was the visual aesthetics; I spent the majority of my development time on this. The difficulty lay in the fact that the visual effects had to match the concept, and to determine whether a particular effect had been achieved, it was necessary to build it and test it repeatedly.

I had originally intended to use TouchDesigner, but as I wasn’t entirely clear on some of its technical concepts, I judged that I wouldn’t have time to make it reliable, so I made a conscious decision *not* to use it, opting instead to create the visual effects using Python/OpenCV. [This was indeed a trade-off: I sacrificed the polished experience of TouchDesigner in favour of a system I truly understood and could debug under pressure.] The visual language went through many iterations; each one taught me something, and each had specific reasons for being discarded:

*Thermal colormap* — readable as "scan," but too static on its own.
*Geometric shapes / annotation boxes* — Discarded:the hard geometry felt clinical in the wrong way and cluttered the frame
*ASCII / data characters*— interesting "body-as-data" idea but hard to read over the background and visually noisy.
*Hydra-style flow distortion* (inspired by DJ_Dave / Olivia Jack) — beautiful, but I realised the states stopped being legible; the story flattened into "pretty movement."
*Final: scan-specimen + signal that fractures over time.* The hand is treated as a specimen being scanned, and the thermal image progressively glitches into broken signal as the machine's "confidence" rises and then collapses.

The central theme of this process is a tension that I must constantly balance: ‘readability versus visual richness’. Every time I make the visuals more attractive or denser, the concept becomes harder to grasp; every time I improve readability, it risks looking like a dull presentation. The final version is my attempt to strike a balance between the two—keeping the early stages clean and saving the breakdown of details for the end.
 
# Ethical Considerations
This work is, in essence, an ethical argument presented through form. It illustrates the power dynamics inherent in algorithmic classification: a system that arbitrarily assumes the right to categorise human bodies into distinct categories; when a body does not conform, the system does not reflect on its own model, but instead demands that the body change (displaying the message ‘Please make your gesture clearer’). This mirrors the dangers present in real-world machine learning: classifiers trained on narrow datasets treat anything falling outside their categories as an error, whilst the burden of ‘making oneself readable’ falls on the human, not the system.”
I also had to consider my own dataset honestly. My KNN model is trained only on my hands, in my lighting, performing four gestures that I chose. It is a tiny, biased dataset — and the piece arguably works because it is biased and brittle. Rather than hide this limitation, the work exposes brittleness as its subject. I think there is an honesty in making the limitation visible, rather than pretending to a generality the system does not actually have. This connects to a wider problem in machine learning: systems are often trained on narrow data and then deployed as if universal, and my piece deliberately stages that gap at a small, personal scale.

# Evaluation: Successes and Failures
*What didn't / what I compromised.*
Because I spent so much of my time iterating on the visual language, other dimensions stayed thinner than I wanted — especially narrative and interaction depth. The piece reads more as a sequence of escalating states than as a fully shaped story. The in-between states (EXPAND, INTERPRETED) are still harder for an audience to tell apart than the extremes (STABLE vs. ERROR). And because the dataset is small and trained only on my own hands in specific lighting, recognition depends on conditions matching my recording session — a fragility I chose to expose rather than solve, but a fragility nonetheless. I should distinguish here between a technical limitation (the small dataset) and a conceptual one (the thin narrative): the first I embraced as part of the work's meaning; the second is a genuine shortfall.

*What I would do with more time.*
My main ambition would be to develop the piece into a more narrative, complete performance — to give it a fuller story arc and a richer, more interactive audience experience, rather than a progression of states. I would want the machine's misreading to unfold like a dramaturgy, with build, turn and consequence, so that the audience is carried through an experience rather than shown a demonstration. I would also widen the gesture vocabulary and test the system with other people's hands, both to strengthen recognition and to deepen the very question the work asks — whose body gets to be "readable"?

# Tools and Attribution
- MediaPipe Hand Landmarker (Google) — hand tracking
- scikit-learn — KNN classifier
- Dorothy (CCI) — performance/audio framework
- Stable Diffusion via diffusers (Hugging Face) — generative hallucination layer
- Sound files from Freesound

# Which parts have I implemented
1.tracking hands in real time with MediaPipe;
2.classifying trained gestures with KNN;
3.mapping gesture states to five machine-reading states;
4.controlling visual intensity through hand distance and movement;
5.generating Stable Diffusion img2img body-scan visuals;
6.creating thermal camera / biometric scan effects;
7.translating the body into ASCII and data-like imagery;
8.triggering diagnostic machine text;
9.controlling particles, echoes, glitch, and tearing effects;
10.changing sound through gesture states and performance tension;
11.presenting the body as data that is overinterpreted by a machine.

# Audience Feedback
After the performance I gathered informal feedback from people who watched it. The clearest response was that the visuals were strong and felt complete as a whole, though some viewers noted that one short section was a little unclear to follow. The strongest positive reaction came in the later part of the performance: as the Stable Diffusion imagery intensified and the random diagnostic text multiplied, several viewers found this the most compelling and exciting moment — the point where the machine's "over-reading" became most vivid. This feedback was useful in two ways. It confirmed that the breakdown and over-interpretation stages communicated most powerfully, which is consistent with my own sense that the failure state is where the piece feels most alive. At the same time, the comment about the unclear section told me that the legibility of the in-between states still needs work, and reinforced the tension between readability and visual richness that I describe above.

# How LLM tools supported 
A Dorothy API limitation with line drawing. dot.line() did not accept a width argument; the fix was to set stroke weight separately (dot.set_stroke_weight()) before drawing.

Frame-rate / performance problems. Heavy per-pixel image operations dropped the frame rate to ~4 FPS. I learned to do the expensive image maths at reduced resolution and upscale, recovering a usable frame rate.


# Reference
Akten, M., Fiebrink, R., & Grierson, M. (2019). Learning to see. ACM SIGGRAPH 2019 Art Gallery on - SIGGRAPH ’19. https://doi.org/10.1145/3306211.3320143

Fiebrink, R., & Caramiaux, B. (2016). The Machine Learning Algorithm as Creative Musical Tool. ArXiv.org. https://arxiv.org/abs/1611.00379

Wikipedia Contributors. (2025, May 25). Zach Blas. Wikipedia; Wikimedia Foundation.

‌Metz, C. (2017, April 30). The Unsettling Performance That Showed the World Through AI’s Eyes. WIRED. https://www.wired.com/2017/04/unsettling-performance-showed-world-ais-eyes/

‌Wikipedia Contributors. (2026, May 29). Hito Steyerl. Wikipedia; Wikimedia Foundation.

‌Caramiaux, B., & Donnarumma, M. (2020). Artificial Intelligence in Music and Performance: A Subjective Art-Research Inquiry. ArXiv.org. https://arxiv.org/abs/2007.15843
‌

‌