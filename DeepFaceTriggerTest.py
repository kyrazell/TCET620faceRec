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

imgLoc = "imgDataCount"
imgPaths = os.listdir(imgLoc)
embeddings = []

for i in range(0,len(imgPaths)):
    imgPaths[i] = imgLoc + "\\" + imgPaths[i]
    embedding_obj = DeepFace.represent(img_path = imgPaths[i])[0]['embedding']
    embeddings.append(embedding_obj)
    
print(imgPaths)


## Initialize list to identify if any of the faces captured in the frame match a face in database
adminFace = [""]

## Initialize selected camera and camera window
liveCam = cv2.VideoCapture(0)

cv2.namedWindow("Test")                                             ## Initialize display window; Unnecessary in final implementation unless live video display would be interrupted otherwise

## Run face verification against each face in admin database
while True:
    ret, frame = liveCam.read()                                     ## Test if cv2/camera supports multi-output (May interrupt live video output if camera 0 is seized)
    
    cv2.imshow("Test", frame)                                       ## Show display window; Unnecessary in final implementation unless live video display would be interrupted otherwise

    k = cv2.waitKey(1)                                              ## Initialize trigger; Unnecessary in final implementatiion assuming full script is run on trigger (gesture)

    if k == ord('q') :
        exit()

    if k == ord('c'):                                               ## Designate trigger, currently letter c on keyboard; Unnecessary in final implementatiion assuming full script is run on trigger
        testImg = "Test.jpg"                                        ## Define name for captured frame
        cv2.imwrite(testImg, frame)                                 ## Capture frame
        try:
            embedTest = DeepFace.represent(img_path = testImg)[0]['embedding']
        except ValueError:
            embedTest = []
            print("Please retake image")
        for j in range(0,len(imgPaths)):                            ## Loop through each admin face and test if any faces captured on frame match admin
            try:
                result = DeepFace.verify(img1_path = embeddings[j], 
                            img2_path = embedTest, 
                            distance_metric = metrics[1],
                            model_name = models[0],
                            silent = True
                )
                if result['verified'] == True :
                    adminTest = imgPaths[j]                         ## If any of the admin faces match, adminFace is updated
                    adminFace.append(adminTest[13:-4])
                print(j)
            except ValueError:                                      ## If no face is detected in frame, return false
                adminFace.append("No Face Detected")
        if len(adminFace)== 1 :
            adminFace.append("Stranger")                            ## if no admin faces match, adminFace is updated with Stranger
        faceRecognized = adminFace[1]
        adminFace = [""]                                            ## Reset adminFace for next capture
        print(faceRecognized)                                       ## Print names of faces recognized
