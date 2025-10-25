import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions import drawing_utils  # We'll need this to draw the landmarks

import cv2 as cv
import time  # Need this for timestamps

# Import MediaPipe Model
mediaPipe_model_path = './Models/hand_landmarker.task'

# Crete Mediapipe Task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# --- CHANGED: Removed the print_result callback function ---
# We will get the result directly from the detect_for_video function

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediaPipe_model_path),
    # --- CHANGED: Use VIDEO mode ---
    running_mode=VisionRunningMode.LIV,
    # --- CHANGED: Removed result_callback ---
)

# Use 'with' statement for resource management
with HandLandmarker.create_from_options(options) as landmarker:
    print("Hand landmarker initialized.")

    # Use OpenCV’s VideoCapture to start capturing from the webcam
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    # --- CHANGED: We need a variable for the timestamp ---
    # We will use time.time() instead of incrementing a counter

    while cam.isOpened():
        # Capture frame-by-frame
        isSuccess, frame = cam.read()

        if not isSuccess:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # --- (FIX 1) ---
        # --- CHANGED: Convert from BGR (OpenCV default) to RGB (MediaPipe expected) ---
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert the frame to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # --- (FIX 2) ---
        # --- CHANGED: Get a proper timestamp in milliseconds ---
        timestamp_ms = int(time.time() * 1000)

        # --- CHANGED: Use detect_for_video (synchronous) instead of detect_async ---
        # This function will block and return the detection result directly.
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- (FIX 3) ---
        # --- ADDED: Draw the landmarks on the *original* BGR frame ---
        annotated_frame = frame.copy()  # Make a copy to draw on
        if detection_result.hand_landmarks:
            print(f"Found {len(detection_result.hand_landmarks)} hand(s):")
            # print(detection_result) # This is what you were doing before

            for hand_landmarks in detection_result.hand_landmarks:
                # Draw the landmarks
                drawing_utils.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,  # Pre-defined connections
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        else:
            print("No hands detected.")

        # --- (FIX 4) ---
        # --- ADDED: Display the frame and add a way to quit ---
        cv.imshow('Hand Landmarks', annotated_frame)

        # Wait for 1ms, and if 'q' is pressed, break the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv.destroyAllWindows()
    print("Camera released and windows closed.")