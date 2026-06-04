# Air Drawing Studio (Hand Tracking)

This project lets you draw in the air using a webcam, with smooth hand tracking and an RGB neon light effect.

## Features
- Real-time hand tracking using MediaPipe
- Professional RGB laser strokes with layered neon glow
- True dual-hand drawing: both hands can draw at the same time
- More stable drawing with gesture hysteresis and smoothing filters
- Ring-finger gesture to lock the current RGB color
- Live FPS and tracking-confidence display on screen
- Draw mode: index finger up
- Eraser mode: index + middle fingers up
- Full canvas transform with both hands: move, rotate, zoom
- Clear canvas with keyboard shortcut

## Requirements
- Python 3.9+
- Webcam

## Setup
1. Open a terminal in this folder.
2. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
& .\.venv\Scripts\python.exe air_draw.py
```

## Controls
- Left or right hand, index finger up: Draw (both hands can draw together)
- Left or right hand, index + middle fingers up: Erase
- Ring finger up: Lock current RGB color
- Both hands with index + middle fingers up: Transform full canvas (move + rotate + zoom)
- `F`: Toggle fullscreen
- `C`: Clear canvas
- `Q` or `Esc`: Quit

## Tips
- Use good room lighting
- Keep your hand in frame
- Keep your palm facing the camera during gestures for best stability
- Stand away from very busy backgrounds for cleaner tracking
