import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import pygetwindow as gw
import win32gui
import win32con
from collections import deque
import sys
import math

# Keep PyAutoGUI fail-safe enabled and set up safe bounds
pyautogui.PAUSE = 0.001

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Camera and screen setup
CAM_WIDTH, CAM_HEIGHT = 640, 360
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

screen_width, screen_height = pyautogui.size()

# Virtual trackpad setup with buffer zones
PHYSICAL_TRACKPAD_WIDTH, PHYSICAL_TRACKPAD_HEIGHT = 300, 180
VIRTUAL_TRACKPAD_WIDTH, VIRTUAL_TRACKPAD_HEIGHT = 400, 280  # 50px buffer on each side

# Physical trackpad boundaries
pad_left = (CAM_WIDTH - PHYSICAL_TRACKPAD_WIDTH) // 2
pad_top = (CAM_HEIGHT - PHYSICAL_TRACKPAD_HEIGHT) // 2
pad_right = pad_left + PHYSICAL_TRACKPAD_WIDTH
pad_bottom = pad_top + PHYSICAL_TRACKPAD_HEIGHT

# Virtual trackpad boundaries (extends beyond physical)
virtual_pad_left = pad_left - 50
virtual_pad_top = pad_top - 50
virtual_pad_right = pad_right + 50
virtual_pad_bottom = pad_bottom + 50

# Enhanced smoothing and movement parameters
SMOOTHING = 0.15
ACCELERATION_FACTOR = 2.5

# Improved pinch detection with angle compensation
PINCH_THRESHOLD = 0.06  # Increased for better angle tolerance
DOUBLE_PINCH_WINDOW = 0.8
PINCH_DEBOUNCE_TIME = 0.1
last_pinch_time = 0
last_release_time = 0
pinch_active = False
pinch_debounce_timer = 0

# Enhanced tracking variables
prev_pos = None
smoothed_pos = None
cursor_velocity = np.array([0.0, 0.0])

# Restored stability mode settings
STABILITY_THRESHOLD = 0.003
STABLE_FRAME_COUNT = 8
movement_history = deque(maxlen=STABLE_FRAME_COUNT)
stability_counter = 0
MIN_STABLE_FRAMES = 3

# Enhanced cursor lock with position freezing
cursor_locked = False
locked_cursor_pos = (0, 0)
click_position_frozen = False
frozen_cursor_pos = (0, 0)
# --- Add scroll lock state ---
scroll_lock_active = False
scroll_lock_release_time = 0.0  # Time until which cursor remains locked after scroll
SCROLL_LOCK_TIMEOUT = 1.0  # seconds
# --- Scroll-to-cursor transition buffer ---
pre_scroll_cursor_pos = None
cursor_settling = False
settling_start_time = 0.0
SETTLING_DURATION = 0.25  # seconds
settling_frames = 8
settling_frame_count = 0
settling_deadzone = 0.07  # normalized units
settling_smoothing = 0.5  # heavy smoothing
settling_target_pos = None
settling_visual_feedback = ''

# --- Remove all old scroll-related code and variables ---
# (SCROLL_MODE, scroll_origin, scroll_last_pos, scroll_last_time, two-finger logic, etc.)

# --- New Scroll System: 4-Finger Thumb-Controlled Scrolling ---
import threading

SCROLL_DISABLE_DURATION = 1.0  # seconds
SCROLL_BASE_SPEED = 4.0        # base scroll units per frame
SCROLL_SPEED_MULTIPLIER = 2.0  # max multiplier for fast movement
SCROLL_MOMENTUM_DECAY = 0.93   # per frame
SCROLL_DIRECTION_DEBOUNCE = 3  # frames
SCROLL_CONFIDENCE_FRAMES = 3   # frames for stable detection

class ScrollState:
    def __init__(self):
        self.enabled = False
        self.direction = None  # 'up', 'down', or None
        self.speed = 0.0
        self.momentum = 0.0
        self.last_center = None
        self.direction_debounce = 0
        self.confidence = 0
        self.disable_timer = 0.0
        self.disable_until = 0.0
        self.visual_feedback = ''
        self.last_frame_time = time.time()

    def reset(self):
        self.enabled = False
        self.direction = None
        self.speed = 0.0
        self.momentum = 0.0
        self.last_center = None
        self.direction_debounce = 0
        self.confidence = 0
        self.visual_feedback = ''

scroll_state = ScrollState()

# --- Helper function for finger extension detection (move to top) ---
def is_finger_extended(hand, finger_tip_idx, finger_pip_idx):
    """Check if a finger is extended based on landmark positions"""
    tip = hand.landmark[finger_tip_idx]
    pip = hand.landmark[finger_pip_idx]
    # Special handling for thumb (landmark 4 and 3)
    if finger_tip_idx == 4:
        # Thumb: check if tip is far from palm in both x and y (for right hand)
        wrist = hand.landmark[0]
        index_mcp = hand.landmark[5]
        x_diff = tip.x - index_mcp.x
        y_diff = abs(tip.y - wrist.y)
        # Stricter threshold for thumb extension
        return (x_diff > 0.05 and y_diff > 0.10)
    return tip.y < pip.y - 0.02

# --- Helper functions for finger detection ---
def count_extended_fingers(hand):
    # Returns the number of extended fingers (thumb, index, middle, ring, pinky)
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    count = 0
    for tip, pip in zip(tips, pips):
        if is_finger_extended(hand, tip, pip):
            count += 1
    return count

def get_finger_tips(hand):
    # Returns the (x, y) positions of all 5 finger tips
    return [hand.landmark[i] for i in [4, 8, 12, 16, 20]]

def calculate_4_finger_center(hand):
    # Returns the center (x, y) of index, middle, ring, pinky tips
    tips = [8, 12, 16, 20]
    xs = [hand.landmark[i].x for i in tips]
    ys = [hand.landmark[i].y for i in tips]
    zs = [hand.landmark[i].z for i in tips]
    return np.array([np.mean(xs), np.mean(ys), np.mean(zs)])

# --- 2. Improved 3D, angle-invariant click detection ---
def get_angle_invariant_3d_pinch_distance(hand):
    """Calculate true 3D distance between thumb and index tip, with angle compensation and depth validation."""
    index_tip = hand.landmark[8]
    thumb_tip = hand.landmark[4]
    index_pip = hand.landmark[6]
    thumb_pip = hand.landmark[3]
    wrist = hand.landmark[0]
    middle_mcp = hand.landmark[9]

    # 3D Euclidean distance
    tip_distance_3d = math.sqrt(
        (index_tip.x - thumb_tip.x) ** 2 +
        (index_tip.y - thumb_tip.y) ** 2 +
        (index_tip.z - thumb_tip.z) ** 2
    )
    # Depth difference
    depth_difference = abs(index_tip.z - thumb_tip.z)
    DEPTH_THRESHOLD = 0.035  # Tunable: prevents false positives when fingers are at different depths
    if depth_difference > DEPTH_THRESHOLD:
        return 1.0  # Return a large value so pinch is not detected

    # Hand orientation compensation (angle-invariant)
    # Hand plane vector
    hand_vector = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z])
    hand_vector = hand_vector / np.linalg.norm(hand_vector)
    # Finger direction vectors
    index_vector = np.array([index_tip.x - index_pip.x, index_tip.y - index_pip.y, index_tip.z - index_pip.z])
    thumb_vector = np.array([thumb_tip.x - thumb_pip.x, thumb_tip.y - thumb_pip.y, thumb_tip.z - thumb_pip.z])
    if np.linalg.norm(index_vector) > 0:
        index_vector = index_vector / np.linalg.norm(index_vector)
    if np.linalg.norm(thumb_vector) > 0:
        thumb_vector = thumb_vector / np.linalg.norm(thumb_vector)
    # Angle between finger vectors and hand plane
    index_angle = np.arccos(np.clip(np.abs(np.dot(index_vector, hand_vector)), 0, 1))
    thumb_angle = np.arccos(np.clip(np.abs(np.dot(thumb_vector, hand_vector)), 0, 1))
    # Angle compensation factor
    angle_compensation = 1.0 + 0.5 * (1.0 - np.cos(index_angle + thumb_angle))
    # Return compensated 3D distance
    return tip_distance_3d * angle_compensation

def get_virtual_coordinates(finger_x, finger_y):
    """Map finger position from the unified virtual trackpad area to virtual coordinates."""
    # Clamp finger position to the virtual trackpad bounds
    virtual_x = np.clip(finger_x, virtual_pad_left, virtual_pad_right)
    virtual_y = np.clip(finger_y, virtual_pad_top, virtual_pad_bottom)
    return virtual_x, virtual_y

def get_dual_mode_coordinates(virtual_x, virtual_y):
    """Map virtual coordinates linearly to normalized (0-1) coordinates for the entire virtual trackpad area."""
    # Linear mapping for the entire area
    norm_x = (virtual_x - virtual_pad_left) / VIRTUAL_TRACKPAD_WIDTH
    norm_y = (virtual_y - virtual_pad_top) / VIRTUAL_TRACKPAD_HEIGHT
    # Clamp to valid range
    norm_x = np.clip(norm_x, 0.0, 1.0)
    norm_y = np.clip(norm_y, 0.0, 1.0)
    return norm_x, norm_y

def apply_stability_filtering(current_pos, target_pos, movement_history):
    """Apply intelligent stability filtering to ignore micro-movements"""
    if current_pos is None:
        return target_pos
    
    # Calculate movement vector
    movement = target_pos - current_pos
    movement_magnitude = np.linalg.norm(movement)
    
    # Store movement for stability analysis
    movement_history.append(movement_magnitude)
    
    # Check if movement is stable (recent movements are small)
    recent_movements = list(movement_history)[-MIN_STABLE_FRAMES:]
    is_stable = len(recent_movements) >= MIN_STABLE_FRAMES and all(m < STABILITY_THRESHOLD for m in recent_movements)
    
    if is_stable:
        # In stable mode: ignore very small movements
        if movement_magnitude < STABILITY_THRESHOLD:
            return current_pos  # Don't update position for micro-movements
        else:
            # Larger movement detected: exit stable mode
            stability_counter = 0
    
    # Apply adaptive smoothing based on movement speed
    if movement_magnitude > 0.1:  # Fast movement
        smoothing_factor = SMOOTHING * 0.5  # Reduced smoothing for responsiveness
    elif movement_magnitude > 0.05:  # Medium movement
        smoothing_factor = SMOOTHING
    else:  # Slow movement
        smoothing_factor = SMOOTHING * 1.5  # Increased smoothing for stability
    
    # Apply smoothing
    smoothed_pos = current_pos + smoothing_factor * movement
    
    # Ensure the smoothed position stays within bounds
    smoothed_pos = np.clip(smoothed_pos, 0.0, 1.0)
    
    return smoothed_pos

# --- 1. Fix virtual trackpad edge mapping ---
def get_screen_position(norm_x, norm_y):
    """Convert normalized coordinates to screen position, ensuring full screen coverage."""
    # Map 0.0 exactly to 0, 1.0 exactly to screen_width-1/screen_height-1
    screen_x = int(round(norm_x * (screen_width - 1)))
    screen_y = int(round(norm_y * (screen_height - 1)))
    # Clamp to valid range
    screen_x = np.clip(screen_x, 0, screen_width - 1)
    screen_y = np.clip(screen_y, 0, screen_height - 1)
    return screen_x, screen_y

def detect_exit_gesture(hand):
    """Detect the exit gesture: index and middle fingers horizontal, moved apart then together"""
    global EXIT_GESTURE_ACTIVE, exit_gesture_start_time
    
    # Check if index and middle fingers are extended
    index_extended = is_finger_extended(hand, 8, 6)
    middle_extended = is_finger_extended(hand, 12, 10)
    
    if index_extended and middle_extended:
        # Get finger positions
        index_tip = hand.landmark[8]
        middle_tip = hand.landmark[12]
        
        # Check if fingers are roughly horizontal (similar Y coordinates)
        y_diff = abs(index_tip.y - middle_tip.y)
        if y_diff < 0.05:  # Fingers are horizontal
            # Check finger separation
            x_diff = abs(index_tip.x - middle_tip.x)
            
            if not EXIT_GESTURE_ACTIVE:
                if x_diff > 0.1:  # Fingers are apart
                    EXIT_GESTURE_ACTIVE = True
                    exit_gesture_start_time = time.time()
            else:
                # Check if fingers are now together
                if x_diff < 0.05:  # Fingers are together
                    if time.time() - exit_gesture_start_time > exit_gesture_duration:
                        return True
                elif x_diff > 0.15:  # Reset if fingers are too far apart
                    EXIT_GESTURE_ACTIVE = False
    else:
        EXIT_GESTURE_ACTIVE = False
    
    return False

def safe_move_cursor(x, y):
    """Safely move cursor with full screen access"""
    try:
        # Allow full screen access including edges
        screen_x = np.clip(x, 0, screen_width)
        screen_y = np.clip(y, 0, screen_height)
        pyautogui.moveTo(screen_x, screen_y, _pause=False)
        return True
    except Exception as e:
        print(f"Cursor movement error: {e}")
        return False

def safe_scroll(amount):
    """Safely perform scrolling"""
    try:
        pyautogui.scroll(int(amount))
        return True
    except Exception as e:
        print(f"Scroll error: {e}")
        return False

# --- Additions for debugging and reliability ---
CLICK_FRAMES_REQUIRED = 3  # Number of consecutive frames to confirm a click
CLICK_DEBOUNCE_TIME = 0.12  # Minimum time between clicks (seconds)
CLICK_LOCK_TIMEOUT = 1.0  # Max time to keep cursor locked (seconds)

# --- Exit gesture state (for exit gesture detection) ---
EXIT_GESTURE_ACTIVE = False
exit_gesture_start_time = 0
exit_gesture_duration = 0.7  # seconds, adjust as needed

click_frame_counter = 0
click_confidence = 0.0
click_last_time = 0
click_lock_start_time = 0

# --- Patch main loop for debug and fixes ---
# Main tracking loop
# --- 4-finger thumb-controlled scrolling system ---
# NEW: Position-based, constant-speed, context-aware thumb brake

# --- Helper for finger extension (distance from tip to wrist) ---
def get_finger_extension(hand, tip_idx, wrist_idx=0):
    tip = hand.landmark[tip_idx]
    wrist = hand.landmark[wrist_idx]
    return math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2 + (tip.z - wrist.z) ** 2)

def get_4_finger_extension_level(hand):
    # Index, middle, ring, pinky
    tips = [8, 12, 16, 20]
    wrist = 0
    extensions = [get_finger_extension(hand, tip, wrist) for tip in tips]
    # Normalize: min = closed, max = fully open (calibrate empirically)
    # Use dynamic min/max for robustness
    min_ext, max_ext = 0.04, 0.10  # These may need tuning for your camera/hand size
    avg_ext = np.mean(extensions)
    level = (avg_ext - min_ext) / (max_ext - min_ext)
    return np.clip(level, 0.0, 1.0), extensions

def is_thumb_extended(hand):
    # Thumb is extended if tip is far from palm in x direction (for right hand)
    # More robust: compare thumb tip to index MCP (landmark 5)
    thumb_tip = hand.landmark[4]
    index_mcp = hand.landmark[5]
    wrist = hand.landmark[0]
    # For right hand, thumb.x > index_mcp.x when extended (flipped for left hand)
    # Use y distance as fallback for vertical hands
    x_diff = thumb_tip.x - index_mcp.x
    y_diff = abs(thumb_tip.y - wrist.y)
    # Relaxed: thumb is extended if x_diff > 0.012 (right hand) or y_diff > 0.06
    return x_diff > 0.012 or y_diff > 0.06

def is_thumb_poked_out(hand):
    thumb_tip = hand.landmark[4]
    wrist = hand.landmark[0]
    palm_indices = [0, 5, 9, 13, 17]
    palm_x = np.mean([hand.landmark[i].x for i in palm_indices])
    palm_y = np.mean([hand.landmark[i].y for i in palm_indices])
    palm_z = np.mean([hand.landmark[i].z for i in palm_indices])
    palm_center = np.array([palm_x, palm_y, palm_z])
    thumb_pos = np.array([thumb_tip.x, thumb_tip.y, thumb_tip.z])
    thumb_to_palm_dist = np.linalg.norm(thumb_pos - palm_center)
    wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
    wrist_to_palm_dist = np.linalg.norm(wrist_pos - palm_center)
    norm_thumb_extension = thumb_to_palm_dist / (wrist_to_palm_dist + 1e-6)
    other_tips = [8, 12, 16, 20]
    other_tips_pos = np.array([[hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z] for i in other_tips])
    avg_other_tips = np.mean(other_tips_pos, axis=0)
    thumb_to_fingers_dist = np.linalg.norm(thumb_pos - avg_other_tips)
    # Stricter thresholds
    EXTENSION_THRESHOLD = 0.75
    FINGER_SEPARATION_THRESHOLD = 0.35
    is_extended = norm_thumb_extension > EXTENSION_THRESHOLD
    is_separated = thumb_to_fingers_dist > FINGER_SEPARATION_THRESHOLD * wrist_to_palm_dist
    # Require both to be true
    return is_extended and is_separated

def get_thumb_status(hand, extension_level, direction):
    # Context-aware thumb status for visual feedback
    if direction == 'up':
        if is_thumb_extended(hand):
            return 'Extended (Brake)'
        else:
            return 'Normal'
    elif direction == 'down':
        if is_thumb_poked_out(hand):
            return 'Poked Out (Brake)'
        else:
            return 'Tucked'
    else:
        return 'Normal'

# --- New Scroll State ---
class NewScrollState:
    def __init__(self):
        self.active = False
        self.direction = None  # 'up', 'down', or None
        self.disable_until = 0.0
        self.debounce_counter = 0
        self.last_extension = None
        self.stable_frames = 0
        self.visual_feedback = ''
        self.finger_count = 0
        self.thumb_status = 'Normal'
        self.extension_level = 0.0

    def reset(self):
        self.active = False
        self.direction = None
        self.debounce_counter = 0
        self.stable_frames = 0
        self.visual_feedback = ''
        self.thumb_status = 'Normal'
        self.extension_level = 0.0

new_scroll_state = NewScrollState()

# --- Configurable parameters ---
SCROLL_CONSTANT_SPEED = 20  # lines per frame
EXTENSION_UP_THRESHOLD = 0.75
EXTENSION_DOWN_THRESHOLD = 0.25
SCROLL_STABLE_FRAMES = 3
THUMB_BRAKE_DISABLE_TIME = 1.0  # seconds

# --- Main tracking loop (replace old scroll logic) ---
# --- Scroll timing and batching variables ---
SCROLL_INTERVAL = 0.07  # seconds between scroll actions (~14 Hz)
SCROLL_MOMENTUM_STEP = 1.0  # how much to increase per interval
SCROLL_MOMENTUM_MAX = 40.0  # max scroll per interval
SCROLL_MOMENTUM_MIN = 10.0  # min scroll per interval
SCROLL_MOMENTUM_DECAY = 0.85  # decay factor when not scrolling
last_scroll_time = time.time()
pending_scroll_amount = 0.0
scroll_momentum = SCROLL_MOMENTUM_MIN
scroll_last_direction = None
scroll_active_last_frame = False
scroll_last_extension_level = None
# --- Main tracking loop ---
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw trackpad areas
        # Physical trackpad (green)
        cv2.rectangle(frame, (pad_left, pad_top), (pad_right, pad_bottom), (0, 255, 0), 2)
        cv2.putText(frame, "Physical Trackpad", (pad_left + 10, pad_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Virtual trackpad boundaries (blue, dashed)
        cv2.rectangle(frame, (virtual_pad_left, virtual_pad_top), (virtual_pad_right, virtual_pad_bottom), (255, 0, 0), 1)
        cv2.putText(frame, "Virtual Trackpad", (virtual_pad_left + 10, virtual_pad_top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            # Always define finger tips before use
            index_tip = hand.landmark[8]
            thumb_tip = hand.landmark[4]
            middle_tip = hand.landmark[12]
            # --- Print thumb tip coordinates on Enter key press for calibration ---
            if cv2.waitKey(1) == 13:  # 13 is Enter key
                print(f"Thumb tip coordinates: x={thumb_tip.x:.3f}, y={thumb_tip.y:.3f}, z={thumb_tip.z:.3f}")
            # --- New 4-finger scroll system ---
            now = time.time()
            # Count extended fingers (for feedback)
            tips = [4, 8, 12, 16, 20]
            pips = [3, 6, 10, 14, 18]
            extended = [is_finger_extended(hand, tip, pip) for tip, pip in zip(tips, pips)]
            extended_count = sum(extended)
            new_scroll_state.finger_count = extended_count

            # --- Scroll timing and batching logic ---
            scroll_triggered = False
            scroll_direction = None
            scroll_amount_this_frame = 0.0
            # Only consider scroll if exactly 4 fingers extended (ignore thumb for now)
            extension_level, _ = get_4_finger_extension_level(hand)
            new_scroll_state.extension_level = extension_level
            # Determine scroll direction based on extension level
            if now < new_scroll_state.disable_until:
                new_scroll_state.active = False
                new_scroll_state.visual_feedback = 'SCROLL DISABLED (BRAKE)'
                scroll_direction = None
            elif new_scroll_state.finger_count == 4:
                # Debounce: require stable extension for a few frames
                direction = None
                if extension_level >= EXTENSION_UP_THRESHOLD:
                    direction = 'up'
                elif extension_level <= EXTENSION_DOWN_THRESHOLD:
                    direction = 'down'
                else:
                    if extension_level - EXTENSION_DOWN_THRESHOLD < EXTENSION_UP_THRESHOLD - extension_level:
                        direction = 'down'
                    else:
                        direction = 'up'
                # Stability check
                if new_scroll_state.direction == direction:
                    new_scroll_state.stable_frames += 1
                else:
                    new_scroll_state.stable_frames = 1
                    new_scroll_state.direction = direction
                # Only scroll if stable
                if new_scroll_state.stable_frames >= SCROLL_STABLE_FRAMES:
                    # Thumb brake logic (INVERTED)
                    thumb_brake = False
                    if direction == 'up' and not is_thumb_extended(hand):
                        thumb_brake = True
                    if direction == 'down' and not is_thumb_poked_out(hand):
                        thumb_brake = True
                    new_scroll_state.thumb_status = get_thumb_status(hand, extension_level, direction)
                    if thumb_brake:
                        new_scroll_state.active = False
                        new_scroll_state.disable_until = now + THUMB_BRAKE_DISABLE_TIME
                        new_scroll_state.visual_feedback = f'SCROLL STOPPED (THUMB BRAKE)'
                        scroll_direction = None
                    else:
                        new_scroll_state.active = True
                        new_scroll_state.visual_feedback = f'SCROLL {direction.upper()} (MOM {int(scroll_momentum)})'
                        scroll_direction = direction
                        scroll_triggered = True
                else:
                    new_scroll_state.active = False
                    new_scroll_state.visual_feedback = 'SCROLL WAITING (STABILIZING)'
                    scroll_direction = None
            elif new_scroll_state.finger_count == 0:
                direction = 'down'
                if new_scroll_state.direction == direction:
                    new_scroll_state.stable_frames += 1
                else:
                    new_scroll_state.stable_frames = 1
                    new_scroll_state.direction = direction
                if new_scroll_state.stable_frames >= SCROLL_STABLE_FRAMES:
                    thumb_brake = False
                    if is_thumb_poked_out(hand):
                        thumb_brake = True
                    new_scroll_state.thumb_status = get_thumb_status(hand, 0.0, direction)
                    if thumb_brake:
                        new_scroll_state.active = False
                        new_scroll_state.disable_until = now + THUMB_BRAKE_DISABLE_TIME
                        new_scroll_state.visual_feedback = f'SCROLL STOPPED (THUMB BRAKE)'
                        scroll_direction = None
                    else:
                        new_scroll_state.active = True
                        new_scroll_state.visual_feedback = f'SCROLL DOWN (MOM {int(scroll_momentum)})'
                        scroll_direction = direction
                        scroll_triggered = True
            else:
                if new_scroll_state.finger_count == 0 and not is_thumb_extended(hand):
                    pass
                else:
                    new_scroll_state.active = False
                    new_scroll_state.stable_frames = 0
                    new_scroll_state.visual_feedback = 'SCROLL INACTIVE (FINGER COUNT)'
                    new_scroll_state.thumb_status = 'Normal'
                scroll_direction = None
            # --- Scroll batching and momentum ---
            if scroll_triggered and scroll_direction:
                # If direction is same as last, increase momentum
                if scroll_last_direction == scroll_direction:
                    scroll_momentum = min(scroll_momentum + SCROLL_MOMENTUM_STEP, SCROLL_MOMENTUM_MAX)
                else:
                    scroll_momentum = SCROLL_MOMENTUM_MIN
                scroll_last_direction = scroll_direction
                # Accumulate scroll amount
                if scroll_direction == 'up':
                    pending_scroll_amount += scroll_momentum
                elif scroll_direction == 'down':
                    pending_scroll_amount -= scroll_momentum
                # --- Enable scroll lock during scrolling ---
                if not scroll_lock_active:
                    pre_scroll_cursor_pos = pyautogui.position()
                scroll_lock_active = True
                scroll_lock_release_time = 0.0  # Reset release timer while scrolling
            else:
                # Decay momentum if not scrolling
                scroll_momentum = max(scroll_momentum * SCROLL_MOMENTUM_DECAY, SCROLL_MOMENTUM_MIN)
                scroll_last_direction = None
                # --- Disable scroll lock when not scrolling ---
                if scroll_lock_active:
                    # Just exited scroll mode, start settling
                    cursor_settling = True
                    settling_start_time = time.time()
                    settling_frame_count = 0
                    settling_visual_feedback = 'CURSOR SETTLING'
                    # Save the last smoothed_pos as the target
                    settling_target_pos = smoothed_pos.copy() if smoothed_pos is not None else None
                scroll_lock_active = False
                if scroll_lock_release_time == 0.0:
                    scroll_lock_release_time = time.time() + SCROLL_LOCK_TIMEOUT
            # Only send scroll command at limited rate
            if abs(pending_scroll_amount) >= 1.0 and (now - last_scroll_time) >= SCROLL_INTERVAL:
                pyautogui.scroll(int(pending_scroll_amount))
                last_scroll_time = now
                pending_scroll_amount = 0.0

            # --- Visual feedback for new scroll state ---
            cv2.putText(frame, new_scroll_state.visual_feedback, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
            cv2.putText(frame, f'Fingers: {new_scroll_state.finger_count}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 0), 2)
            cv2.putText(frame, f'Thumb: {new_scroll_state.thumb_status}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f'Extension: {int(new_scroll_state.extension_level * 100)}%', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

            # Check for exit gesture first
            if detect_exit_gesture(hand):
                print("Exit gesture detected! Closing application...")
                break

            # Enhanced pinch detection with angle compensation
            pinch_distance = get_angle_invariant_3d_pinch_distance(hand)
            now = time.time()
            
            # --- Improved click detection with multi-frame validation and confidence ---
            # Adaptive threshold: base + angle compensation
            base_threshold = PINCH_THRESHOLD
            angle_factor = 1.0 + 0.5 * (1.0 - np.cos(np.pi/4))  # Example: adjust for 45deg
            adaptive_threshold = base_threshold * angle_factor
            is_pinch = pinch_distance < adaptive_threshold

            # Multi-frame validation for click
            if is_pinch:
                click_frame_counter += 1
            else:
                click_frame_counter = 0
            click_confidence = min(click_frame_counter / CLICK_FRAMES_REQUIRED, 1.0)

            # Debounce and lock logic
            if click_frame_counter >= CLICK_FRAMES_REQUIRED and now - click_last_time > CLICK_DEBOUNCE_TIME:
                if not pinch_active:
                    pinch_active = True
                    # (do not set cursor_locked or click_position_frozen here)
                    click_lock_start_time = now
                    if now - last_release_time < DOUBLE_PINCH_WINDOW:
                        pyautogui.doubleClick()
                    else:
                        pyautogui.click()
                    last_pinch_time = now
                    pinch_debounce_timer = now
                    click_last_time = now
            elif not is_pinch and pinch_active:
                pinch_active = False
                cursor_locked = False
                click_position_frozen = False
                last_release_time = now
                click_frame_counter = 0
                click_confidence = 0.0

            # Timeout for stuck lock
            if cursor_locked and (now - click_lock_start_time > CLICK_LOCK_TIMEOUT):
                pinch_active = False
                # (do not set cursor_locked or click_position_frozen here)
                click_frame_counter = 0
                click_confidence = 0.0

            # --- Set cursor lock and position freeze based on scroll or click ---
            lock_now = scroll_lock_active or pinch_active
            if lock_now:
                if not (cursor_locked and click_position_frozen):
                    frozen_cursor_pos = pyautogui.position()
                cursor_locked = True
                click_position_frozen = True
            else:
                cursor_locked = False
                click_position_frozen = False
                scroll_lock_release_time = 0.0  # Reset after timeout

            # --- Visual feedback for click confidence and lock ---
            feedback_text = f"CLICK CONF: {click_confidence:.2f}"
            color = (0, 0, 255) if cursor_locked else (0, 255, 0)
            cv2.putText(frame, feedback_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if cursor_locked:
                cv2.putText(frame, "CURSOR LOCKED", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Get finger positions in camera coordinates
            index_x = int(index_tip.x * CAM_WIDTH)
            index_y = int(index_tip.y * CAM_HEIGHT)
            thumb_x = int(thumb_tip.x * CAM_WIDTH)
            thumb_y = int(thumb_tip.y * CAM_HEIGHT)
            
            # Check if fingers are within virtual trackpad area
            if (virtual_pad_left <= index_x <= virtual_pad_right and 
                virtual_pad_top <= index_y <= virtual_pad_bottom):
                
                # Convert to virtual coordinates with buffer zones
                virtual_index_x, virtual_index_y = get_virtual_coordinates(index_x, index_y)
                virtual_thumb_x, virtual_thumb_y = get_virtual_coordinates(thumb_x, thumb_y)
                
                # Apply dual-mode tracking
                norm_x, norm_y = get_dual_mode_coordinates(virtual_index_x, virtual_index_y)
                
                # Apply stability filtering
                target_pos = np.array([norm_x, norm_y])
                smoothed_pos = apply_stability_filtering(smoothed_pos, target_pos, movement_history)
                
                # Calculate screen position
                screen_x, screen_y = get_screen_position(smoothed_pos[0], smoothed_pos[1])

                # --- Cursor lock during click or scroll, and scroll-to-cursor settling ---
                if cursor_locked or click_position_frozen:
                    safe_move_cursor(*frozen_cursor_pos)
                elif cursor_settling and settling_target_pos is not None and smoothed_pos is not None:
                    # Settling: interpolate from pre_scroll_cursor_pos to current index finger position
                    elapsed = time.time() - settling_start_time
                    t = min(1.0, (settling_frame_count + 1) / settling_frames)
                    # Get current index finger normalized position
                    norm_x, norm_y = smoothed_pos[0], smoothed_pos[1]
                    # Calculate deadzone: require finger to move a minimum distance from where it was at scroll end
                    dist = np.linalg.norm(smoothed_pos - settling_target_pos)
                    if elapsed < SETTLING_DURATION or dist < settling_deadzone:
                        # Heavy smoothing
                        interp_pos = settling_target_pos * (1 - t * settling_smoothing) + smoothed_pos * (t * settling_smoothing)
                        screen_x, screen_y = get_screen_position(interp_pos[0], interp_pos[1])
                        safe_move_cursor(screen_x, screen_y)
                        settling_frame_count += 1
                        settling_visual_feedback = 'CURSOR SETTLING'
                    else:
                        # Settling done, resume normal tracking
                        cursor_settling = False
                        settling_visual_feedback = ''
                        screen_x, screen_y = get_screen_position(smoothed_pos[0], smoothed_pos[1])
                        safe_move_cursor(screen_x, screen_y)
                    # Visual feedback
                    cv2.putText(frame, settling_visual_feedback, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
                else:
                    # Normal tracking
                    screen_x, screen_y = get_screen_position(smoothed_pos[0], smoothed_pos[1])
                    safe_move_cursor(screen_x, screen_y)

                # --- Visual feedback for click confidence and lock ---
                feedback_text = f"CLICK CONF: {click_confidence:.2f}"
                color = (0, 0, 255) if cursor_locked else (0, 255, 0)
                cv2.putText(frame, feedback_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if cursor_locked:
                    cv2.putText(frame, "CURSOR LOCKED", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Debug: Print norm_x, norm_y and screen_x, screen_y at edges
                if 'norm_x' in locals() and 'norm_y' in locals():
                    if norm_x > 0.98 or norm_y > 0.98 or norm_x < 0.02 or norm_y < 0.02:
                        print(f"DEBUG: norm_x={norm_x:.3f}, norm_y={norm_y:.3f}, screen_x={screen_x}, screen_y={screen_y}")

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            # Visual feedback for gestures and modes
            if pinch_active:
                cv2.putText(frame, "CLICK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif EXIT_GESTURE_ACTIVE:
                cv2.putText(frame, "EXIT GESTURE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show current mode
            if smoothed_pos is not None:
                norm_x, norm_y = smoothed_pos[0], smoothed_pos[1]
                # Remove edge/precision mode distinction, always show 'TRACKPAD MODE'
                cv2.putText(frame, "TRACKPAD MODE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Hand Trackpad", frame)

        # Keep the OpenCV window always in the foreground
        try:
            win = gw.getWindowsWithTitle("Hand Trackpad")[0]
            if win is not None:
                win32gui.ShowWindow(win._hWnd, win32con.SW_RESTORE)
                win32gui.SetWindowPos(
                    win._hWnd,
                    win32con.HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
                )
        except Exception:
            pass

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
