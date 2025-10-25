import time

import mediapipe as mp
from mediapipe.tasks import python

import cv2 as cv

from Prediction import run_model

# Import MediaPipe Model
mediaPipe_model_path = '../models/hand_landmarker.task'

# Crete Mediapipe Task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if len(result.hand_world_landmarks) == 0 or len(result.handedness) == 0:
        print('invalid')
        return

    x = result.hand_world_landmarks[0][0].x
    y = result.hand_world_landmarks[0][0].y
    z = result.hand_world_landmarks[0][0].y

    confidence = result.handedness[0][0].score
    hand = result.handedness[0][0].category_name

    request = [x, y, z, confidence, hand]

    print(run_model(request))


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediaPipe_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
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

        # Display Video and when 'q' is entered, destroy the window
        cv.imshow('Image', frame)
        if cv.waitKey(1) & 0xff == ord('q'):
            break

        time.sleep(0.5)

    # When everything done, release the capture
    cam.release()
