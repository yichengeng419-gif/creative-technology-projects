#THE RIGHT TO PAUSE

#Project Overview

The Right to Pause is an interactive installation that examines how people change their bodily behaviour when they are observed and evaluated by an algorithmic system.Participants interact with a fictional verification terminal through a brass touch sensor, or a mouse for remote access. The system measures contact duration, interruptions and response patterns; it rejects, accepts and classifies each interaction. Sustained stillness is the only gesture that interrupts it — and the pause it finally records is met with an archive, a counter-letter, and a space where no further proof is asked.

The project asks: how does algorithmic evaluation pressure people to adjust their behaviour — and what remains when they simply stop?

#Interaction Flow

- The participant holds the mouse, Space key or brass touch sensor to wake the dormant specimen.
- The particle body moves into the verification field.
- The system requests a five-second pause.
- The first successful pause is rejected.
- The system increases the requirement to eight seconds.
- The participant decides whether to try again.
- The completed interaction is classified and stored as a session record.
- A system report and personal counter-letter are displayed.

# Current Features

- WebGL particle-based body visualisation
- Mouse, keyboard and MPR121 touch-sensor input
- Five-second and eight-second pause requests
- Timeout and interrupted-attempt handling
- Browser-based session storage
- CSV data export
- Experimental Random Forest classification
- Model and dataset audit panel
- Optional camera-based movement observation
- Reactive audio during the interaction

## Running the Project

The project must be opened through a local web server. Opening `index.html` directly may prevent assets and modules from loading.

From the project directory, run:

```bash
python3 -m http.server 8001

OPEN- http://localhost:8001

