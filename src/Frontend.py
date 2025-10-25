import time

import mediapipe as mp
from mediapipe.tasks import python

import cv2 as cv

from Prediction import run_model

# Import MediaPipe Model
mediaPipe_model_path = '../models/hand_landmarker.task'

HAND_COLORS = [(0, 255, 0), (0, 0, 255)]

# Crete Mediapipe Task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

annotated_frame = None
NUM_HANDS = 1

# Create a hand landmarker instance with the live stream mode
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if len(result.hand_world_landmarks) == 0 or len(result.handedness) == 0:
        print('invalid')
    else:
        x = result.hand_world_landmarks[0][0].x
        y = result.hand_world_landmarks[0][0].y
        z = result.hand_world_landmarks[0][0].y

        confidence = result.handedness[0][0].score
        hand = result.handedness[0][0].category_name

        request = [x, y, z, confidence, hand]

        print(run_model(request))

    hand_display_callback(result, output_image, timestamp_ms)

def hand_display_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms):
    print('hand landmarker result: {}'.format(result))
    global annotated_frame

    frame_display =  cv.cvtColor(output_image.numpy_view(), cv.COLOR_BGR2RGB)
    height, width, _ = frame_display.shape

    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            color = HAND_COLORS[i % len(HAND_COLORS)]
            pixel_coords = [(int((lm.x) * width), int(lm.y * height)) for lm in hand_landmarks]
            # Draw points
            for (x, y) in pixel_coords:
                cv.circle(frame_display, (x, y), 5, color, -1)
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
                cv.line(frame_display, (x1, y1), (x2, y2), color, 2)


    annotated_frame = frame_display

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediaPipe_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=NUM_HANDS
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

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

        time.sleep(0.3)
    # When everything done, release the capture
    cam.release()
