print("Hello World")

# WHAT NEXT?

# DEPENDENCIES
# python -m pip install mediapipe
# DOWNLOAD THIS https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    base_options=BaseOptions(model_asset_path=mediaPipe_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  print("Initialised")

  