import time

import mediapipe as mp
from mediapipe.tasks import python

import cv2 as cv

# Import MediaPipe Model
mediaPipe_model_path = '../models/hand_landmarker.task'

# Crete Mediapipe Task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

results = []

# Create a hand landmarker instance with the live stream mode
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    if len(result.hand_world_landmarks) == 0 or len(result.handedness) == 0:
        return


    x = result.hand_world_landmarks[0][0].x
    y = result.hand_world_landmarks[0][0].y
    z = result.hand_world_landmarks[0][0].y

    confidence = result.handedness[0][0].score
    hand = result.handedness[0][0].category_name

    results.append({
        'x': x,
        'y': y,
        'z': z,
        'confidence': confidence,
        'hand': hand
    })

    print(len(results))

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

    # When everything done, release the capture
    cam.release()


# Write training data to csv
# with open("../data/gesture_data.csv", 'w') as csvfile:
#     csvfile.write("x,y,z,confidence,hand,gesture\n")
#     for line in results:
#        csvfile.write(str(line['x']) + ',' + str(line['y']) + ',' + str(line['z']) + ',' + str(line['confidence']) + ',' + str(line['hand']) + ",thumb_up" + '\n')

# Write training data to csv (invalid data)
with open("../data/gesture_data.csv", 'a') as csvfile:
   for line in results:
       csvfile.write(
           str(line['x']) + ',' + str(line['y']) + ',' + str(line['z']) + ',' + str(line['confidence']) + ',' + str(line['hand']) + ",invalid" + '\n')

"""
    Gestures: 
        THUMBSUP_TEST = auto()
        SHIELD = auto() # Open palm like a stop sign
        BIND = auto()  # Closed fist
        FIREBALL = auto() # Open palm fingers apart like a claw, casting a fireball
"""

"""
hand landmarker result: 

    HandLandmarkerResult(
    
    handedness=[[
        Category(index=1, score=0.6997196078300476, display_name='Left', category_name='Left')
    ]],
    
    # HandCategory = handedness.Category.category_name
    # HandCategoryConfidence = headedness.Category.score
     
    hand_landmarks=[[
        NormalizedLandmark(x=0.8615632057189941, y=1.0679802894592285, z=5.88610760132724e-07, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.7644948363304138, y=1.0436103343963623, z=-0.04517800733447075, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.6924910545349121, y=0.9874093532562256, z=-0.08224716782569885, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.6596720218658447, y=0.9307432174682617, z=-0.11398475617170334, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.6247397661209106, y=0.8947317004203796, z=-0.14901258051395416, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.7132965326309204, y=0.823937714099884, z=-0.09163936972618103, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.6649298667907715, y=0.6864019632339478, z=-0.13594017922878265, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.6402434706687927, y=0.6021814346313477, z=-0.16481482982635498, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.6208546161651611, y=0.5325576663017273, z=-0.18421658873558044, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.778465747833252, y=0.8080796599388123, z=-0.09555533528327942, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.7503835558891296, y=0.6468278169631958, z=-0.1344742327928543, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.7366140484809875, y=0.5473387837409973, z=-0.16185878217220306, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.7267974615097046, y=0.4632219672203064, z=-0.18151122331619263, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.8410376906394958, y=0.8214482069015503, z=-0.1018759086728096, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.8203865885734558, y=0.67049241065979, z=-0.1368916630744934, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.8071565628051758, y=0.5716099739074707, z=-0.16332551836967468, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.7987014055252075, y=0.4910811185836792, z=-0.1817464828491211, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.9053961038589478, y=0.8553563356399536, z=-0.1106046810746193, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.9018255472183228, y=0.7398893237113953, z=-0.1422332376241684, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.8969351053237915, y=0.6608471870422363, z=-0.15915034711360931, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.8917295336723328, y=0.5908070802688599, z=-0.17012932896614075, visibility=0.0, presence=0.0)
    ]], 
        
    hand_world_landmarks=[[
        Landmark(x=0.02490883320569992, y=0.07272148132324219, z=0.05169190838932991, visibility=0.0, presence=0.0), 
        Landmark(x=-0.0056673940271139145, y=0.05773797631263733, z=0.03569285199046135, visibility=0.0, presence=0.0), 
        Landmark(x=-0.023372462019324303, y=0.038286060094833374, z=0.02811845764517784, visibility=0.0, presence=0.0), 
        Landmark(x=-0.03609703481197357, y=0.02117462083697319, z=0.01098322868347168, visibility=0.0, presence=0.0), 
        Landmark(x=-0.04934317246079445, y=0.005178563762456179, z=-0.0057504186406731606, visibility=0.0, presence=0.0), 
        Landmark(x=-0.026250265538692474, y=-0.0026437239721417427, z=0.0028201022651046515, visibility=0.0, presence=0.0), 
        Landmark(x=-0.03526593744754791, y=-0.029482655227184296, z=-0.004688141401857138, visibility=0.0, presence=0.0), 
        Landmark(x=-0.042740993201732635, y=-0.047939129173755646, z=-0.006412277463823557, visibility=0.0, presence=0.0), 
        Landmark(x=-0.05338858440518379, y=-0.06189144402742386, z=-0.01921129785478115, visibility=0.0, presence=0.0), 
        Landmark(x=-0.005487350281327963, y=-0.004341062158346176, z=0.0026427614502608776, visibility=0.0, presence=0.0), 
        Landmark(x=-0.009655607864260674, y=-0.041240040212869644, z=-0.007947500795125961, visibility=0.0, presence=0.0), 
        Landmark(x=-0.020735546946525574, y=-0.06418934464454651, z=-0.017623838037252426, visibility=0.0, presence=0.0), 
        Landmark(x=-0.02864922396838665, y=-0.0849887877702713, z=-0.023779036477208138, visibility=0.0, presence=0.0), 
        Landmark(x=0.014890292659401894, y=0.0004103983810637146, z=-0.0030101255979388952, visibility=0.0, presence=0.0), 
        Landmark(x=0.00999078806489706, y=-0.030394408851861954, z=-0.009995647706091404, visibility=0.0, presence=0.0), 
        Landmark(x=0.001979290973395109, y=-0.05534644052386284, z=-0.02162865549325943, visibility=0.0, presence=0.0), 
        Landmark(x=-0.00434924615547061, y=-0.07539191097021103, z=-0.023912442848086357, visibility=0.0, presence=0.0), 
        Landmark(x=0.03190790116786957, y=0.013186327181756496, z=0.0012070995289832354, visibility=0.0, presence=0.0), 
        Landmark(x=0.03258465602993965, y=-0.011609588749706745, z=-0.004724335856735706, visibility=0.0, presence=0.0), 
        Landmark(x=0.030518513172864914, y=-0.03437051922082901, z=-0.012844347395002842, visibility=0.0, presence=0.0), 
        Landmark(x=0.02553587220609188, y=-0.04751557484269142, z=-0.01778094470500946, visibility=0.0, presence=0.0)'
    ]]
    )
    
    
    Final Labels: x, y, z, hand category (left / right), category confidence
    
"""