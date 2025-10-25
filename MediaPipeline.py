print("Hello World")

# WHAT NEXT?

# DEPENDENCIES
# python -m pip install mediapipe
# DOWNLOAD THIS https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models

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

# Create a hand landmarker instance with the live stream mode
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
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

        # Create a loop to read the latest frame from the camera using VideoCapture#read()
        numpy_frame_from_opencv = frame

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

        # Send live image data to perform hand landmarks detection.
        # The results are accessible via the `result_callback` provided in
        # the `HandLandmarkerOptions` object.
        # The hand landmarker must be created with the live stream mode.
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        print_result(landmarker.result)

    # When everything done, release the capture
    cam.release()