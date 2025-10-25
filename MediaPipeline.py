print("Hello World")

# WHAT NEXT?

# DEPENDENCIES
# python -m pip install mediapipe
# DOWNLOAD THIS https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models

import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv

# Import MediaPipe Model
mediaPipe_model_path = './Models/hand_landmarker.task'

# Crete Mediapipe Task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

annotated_frame = None

# Create a hand landmarker instance with the live stream mode
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

def hand_display_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms):
    print('hand landmarker result: {}'.format(result))
    global annotated_frame

    if output_image is not None:
        frame =  cv.cvtColor(output_image.numpy_view(), cv.COLOR_RGB2BGR)
    else:
        return
    if result.hand_landmarks:
        height, width, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            pixel_coords = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
            # Draw points
            for (x, y) in pixel_coords:
                cv.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            for start, end in connections:
                x1, y1 = pixel_coords[start]
                x2, y2 = pixel_coords[end]
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    annotated_frame = frame

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediaPipe_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=hand_display_callback
)
with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    print("Initialised")

    # Use OpenCV’s VideoCapture to start capturing from the webcam
    cam = cv.VideoCapture(0)  # Open just 1 camera
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    frame_timestamp_ms = 0
    while cam.isOpened():
        frame_timestamp_ms += 1

        # Capture frame-by-frame
        isSuccess, frame = cam.read()

        if not isSuccess: # If frame is read correctly isSuccess is True
          print("Can't receive frame (stream end?). Exiting ...")
          break

        frame = cv.flip(frame, 1)

        # Convert color from BGR into RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Create timstamp
        timestamp_ms = int(time.time() * 1000)

        # Send live image data to perform hand landmarks detection.
        # The results are accessible via the `result_callback` provided in
        # the `HandLandmarkerOptions` object.
        # The hand landmarker must be created with the live stream mode.
        landmarker.detect_async(mp_image, timestamp_ms)

        # Display latest annotated frame
        if annotated_frame is not None:
            cv.imshow('Image', annotated_frame)
        else:
            cv.imshow('Image', frame)

        # Display Video and when 'q' is entered, destroy the window
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    # When everything done, release the capture
    cam.release()