## Import DeepFace for facial recognition
## Import numpy for boolean arrays
## Import cv2 for Image Capture
from deepface import DeepFace
import numpy as np
import cv2
import fnmatch
import os

## Define different model types, metrics and backends available for DeepFace as array for human readability
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

metrics = ["cosine", "euclidean", "euclidean_l2"]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

## Define image paths for all admin faces (Increasing pictures significantly affects computation time)
## Currently hardcoded in, can easily be modified to for loop for all images in directory (see DeepFaceTriggerTrue.py) 
imgPaths = [
  "imgDataNamed\Kriz.jpg", 
  "imgDataNamed\MattB.jpg", 
  "imgDataNamed\DJ.jpg", 
  "imgDataNamed\Brayden.jpg", 
]

#### Ignore
#### Test Code being worked on to use face embeddings directly for verification to lighten computational load/alleviate freezing
####
####  Convert all images in admin face database to embeddings
##for i in range(0,len(imgPaths)):
##    embedding_objs = DeepFace.represent(img_path = imgPaths[i])

## Initialize boolean matrix to identify if any of the faces captured in the frame match an admin face
adminStatus = [False]

## Initialize selected camera and camera window
liveCam = cv2.VideoCapture(0)

cv2.namedWindow("Test")                                             ## Initialize display window; Unnecessary in final implementation unless live video display would be interrupted otherwise

## Run face verification against each face in admin database
while True:
    ret, frame = liveCam.read()                                     ## Test if cv2/camera supports multi-output (May interrupt live video output if camera 0 is seized)
    
    cv2.imshow("Test", frame)                                       ## Show display window; Unnecessary in final implementation unless live video display would be interrupted otherwise

    k = cv2.waitKey(1)                                              ## Initialize trigger; Unnecessary in final implementatiion assuming full script is run on trigger (gesture)
    if k == ord('c'):                                               ## Designate trigger, currently letter c on keyboard; Unnecessary in final implementatiion assuming full script is run on trigger
        testImg = "Test.jpg"                                        ## Define name for captured frame
        cv2.imwrite(testImg, frame)                                 ## Capture frame
        for j in range(0,len(imgPaths)):                            ## Loop through each admin face and test if any faces captured on frame match admin
            try:
                result = DeepFace.verify(img1_path = imgPaths[j], 
                          img2_path = "Test.jpg", 
                          distance_metric = metrics[1],
                          model_name = models[0]    
                )
                adminStatus.append(result['verified'])              ## Print boolean verified result to adminStatus boolean array
            except ValueError:                                      ## If no face is detected in frame, return false
                adminStatus.append(False)
        adminVerify = max(adminStatus)                              ## if any of the admin faces match, adminVerify returns true (Check DeepFaceTriggerTrue.py)
        print(adminVerify)                                          ## Print boolean value of if admin or not
