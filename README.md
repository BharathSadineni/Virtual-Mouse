# Virtual Mouse with Hand Tracking

A Python-based virtual mouse/trackpad system that uses your webcam and hand gestures to control the mouse cursor, perform clicks, and scroll on your Windows computer. Powered by OpenCV, MediaPipe, and PyAutoGUI, this project enables touchless, intuitive control of your computer using only your hand.

## Features

- **Hand Tracking:** Uses MediaPipe to detect and track your hand in real-time.
- **Virtual Trackpad:** Maps a region in your webcam feed to your screen, allowing you to move the mouse cursor by moving your index finger.
- **Click Detection:** Pinch gesture (thumb and index finger together) triggers a mouse click. Double pinch triggers a double-click.
- **Scrolling:**
  - **Scroll Up:** Extend four fingers (excluding thumb) and use thumb position to scroll up.
  - **Scroll Down:** **Close your fist** (all fingers folded) to scroll down.
  - **Thumb Brake:** Use thumb position to stop scrolling (see below).
- **Edge-to-Edge Mapping:** Full screen coverage, including screen edges and corners.
- **Stability Filtering:** Intelligent smoothing and micro-movement filtering for stable cursor control.
- **Exit Gesture:** Special two-finger gesture to safely exit the application.
- **Visual Feedback:** On-screen overlays show current mode, gesture confidence, and system state.
- **Always-on-Top Window:** The OpenCV window remains visible and on top for easy monitoring.

## Requirements

- **OS:** Windows 10 or later
- **Python:** 3.11 (recommended)
- **Webcam:** Any standard webcam

### Python Dependencies

All dependencies are listed in `requirements.txt`.

> Install them with:
> ```bash
> pip install -r requirements.txt
> ```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/virtual-mouse.git
   cd virtual-mouse
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   ```bash
   python hand_tracking.py
   ```

2. **A window titled "Hand Trackpad" will appear.**
   - The green rectangle is your "physical" trackpad area.
   - The blue rectangle is the "virtual" trackpad (with buffer zones for easier edge access).

3. **Control the mouse:**
   - Move your index finger within the virtual trackpad area to move the cursor.
   - Pinch (thumb and index finger together) to click.
   - Double pinch (two quick pinches) to double-click.
   - Extend four fingers (excluding thumb) and use thumb position to scroll up.
   - **Close your fist (all fingers folded) to scroll down.**
   - Use the special exit gesture to close the app.

## Gestures & Controls

| Gesture                                 | Action                        |
|------------------------------------------|-------------------------------|
| Index finger moves                       | Move mouse cursor             |
| Pinch (thumb + index)                    | Mouse click                   |
| Double pinch                             | Double-click                  |
| Four fingers extended (no thumb)         | Enable scroll up mode         |
| **Fist (all fingers folded)**            | **Scroll down**               |
| Thumb extended (up)                      | Scroll up (release to brake)  |
| Thumb poked out (down)                   | Scroll down (release to brake)|
| Two-finger horizontal apart/together     | Exit application              |

### Visual Feedback

- **"CLICK"**: Pinch detected, click triggered.
- **"CURSOR LOCKED"**: Cursor is locked during click or scroll.
- **"SCROLL UP/DOWN"**: Scrolling in progress.
- **"SCROLL STOPPED (THUMB BRAKE)"**: Scrolling paused by thumb brake.
- **"EXIT GESTURE"**: Exit gesture detected, app will close.

## Troubleshooting

- **Cursor not moving?**
  - Ensure your hand is within the virtual trackpad area (blue rectangle).
  - Good lighting and a clear background improve detection.
- **Clicks not registering?**
  - Try pinching more clearly; adjust your hand angle.
- **Scrolling not working?**
  - Extend four fingers (excluding thumb) for scroll up, **close your fist for scroll down**.
- **App not closing?**
  - Use the two-finger horizontal apart/together gesture, or press `Esc` to exit.

## Customization

- **Trackpad Size:** Adjust `PHYSICAL_TRACKPAD_WIDTH`, `PHYSICAL_TRACKPAD_HEIGHT`, `VIRTUAL_TRACKPAD_WIDTH`, and `VIRTUAL_TRACKPAD_HEIGHT` in `hand_tracking.py`.
- **Gesture Sensitivity:** Tune thresholds like `PINCH_THRESHOLD`, `EXTENSION_UP_THRESHOLD`, and `EXTENSION_DOWN_THRESHOLD`.
- **Smoothing/Responsiveness:** Modify `SMOOTHING`, `STABILITY_THRESHOLD`, and related parameters.

## Known Limitations

- May require calibration for different lighting/camera setups.

---

**Enjoy touchless control of your PC!** 
