# Virtual Mouse with Hand Tracking

A Python-based virtual mouse/trackpad system that uses your webcam and hand gestures to control the mouse cursor, perform clicks, and scroll on your Windows computer. Powered by OpenCV, MediaPipe, and more.

## Demo Video of Virtual Mouse
<p align="center">
  <a href="https://bharathsadineniportfolio.netlify.app/static/media/Virtual%20Mouse%20Video.2623788318840cbf7eab.mp4">
    <img src="https://github.com/BharathSadineni/Virtual-Mouse/blob/main/Virtual%20Mouse%20Demo%20Image.png" alt="Demo Video Screenshot">
  </a>
  <br>
  <em>Click the image above to watch the demo video.</em>
</p>

---

## Features

- Control mouse cursor using hand gestures via webcam
- Single and double click support
- Drag and drop gesture
- Scroll up/down using finger gestures
- Works on Windows
- Real-time feedback on hand detection

## How it Works

The system uses a webcam feed and leverages MediaPipe for real-time hand tracking. It maps finger positions and gestures to mouse actions using OpenCV for image capture and processing, and PyAutoGUI for OS-level mouse control.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BharathSadineni/Virtual-Mouse.git
   cd Virtual-Mouse
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include:
   - opencv-python
   - mediapipe
   - pyautogui
   - numpy

3. *(Optional)* If your system has multiple screens or DPI issues, adjust screen resolution settings in the code.

## Usage

1. Run the main script:
   ```bash
   python VirtualMouse.py
   ```
2. Ensure your webcam is connected and allow access.
3. Use your index finger to move the cursor.
4. Pinch your index and middle finger for clicks and gestures.

## Hand Gestures

| Gesture                      | Action         |
|------------------------------|---------------|
| Index finger up              | Move cursor   |
| Index & middle finger up     | Ready to click|
| Pinch index & thumb          | Left click    |
| Pinch index & pinky          | Right click   |
| Pinch with fingers + move    | Drag/Scroll   |

> The gestures are displayed visually in the webcam window for guidance.

## Troubleshooting

- If the mouse control is laggy, reduce webcam resolution or increase lighting.
- If gestures are not recognized, adjust your hand distance to the camera.
- For multi-monitor setups, ensure the correct screen resolution is set in the code.

## Credits

- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)
