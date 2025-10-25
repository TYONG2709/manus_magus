print("Hello World")

# WHAT NEXT?

# DEPENDENCIES
# python -m pip install mediapipe
# DOWNLOAD THIS https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '/absolute/path/to/gesture_recognizer.task'