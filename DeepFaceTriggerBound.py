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

## Define image paths for all admin faces (Increasing pictures significantly affects pre-computation time)

imgLoc = "imgDataCount"
imgPaths = os.listdir(imgLoc)
embeddings = []

for i in range(0,len(imgPaths)) :
    imgPaths[i] = imgLoc + "\\" + imgPaths[i]
    embedding_obj = DeepFace.represent(img_path = imgPaths[i])[0]['embedding']
    embeddings.append(embedding_obj)
    
print(imgPaths)


## Initialize list to identify if any of the faces captured in the frame match a face in database
adminFace = [""]

## Initialize selected camera and camera window
liveCam = cv2.VideoCapture(0)

cv2.namedWindow("Feed")                                                                         ## Initialize display window

## Run face verification against each face in admin database
while True:
    ret, frame = liveCam.read()                                                                 ## Initialize capture
    
    cv2.imshow("Feed", frame)                                                                   ## Show display window
    
    k = cv2.waitKey(1)                                                                          ## Initialize trigger

    if k == ord('q') :                                                                          ## Designate trigger for quitting, currently letter q
        exit()

    if k == ord('e') :                                                                          ## Designate trigger for embedding, currently letter e
        embedImg = imgLoc + "\\" + input("Please input name: ") + ".jpg"                        ## Prompt name for individual being embedded
        cv2.imwrite(embedImg, frame)                                                            ## Capture frame
        try:
            embedTest = DeepFace.represent(img_path = embedImg)[0]['embedding']
            embeddings.append(embedTest)
            imgPaths.append(embedImg)
        except ValueError :
            embedTest = []
            print("Please retake image")

    if k == ord('c') :                                                                          ## Designate trigger for capturing, currently letter c 
        testImg = "Test.jpg"                                                                    ## Define name for captured frame
        cv2.imwrite(testImg, frame)                                                             ## Capture frame

        try :
            embedRes = DeepFace.represent(img_path = testImg)
            prevArea = 0
            biggestBound = 0
            for m in range(0, len(embedRes)) :                                                  ## Check for which face is most prominent in frame
                faceArea = embedRes[m]['facial_area']['w'] * embedRes[m]['facial_area']['h']
                if faceArea > prevArea :
                    biggestBound = m
                prevArea = faceArea
            embedTest = embedRes[biggestBound]['embedding']
        except ValueError :
            embedTest = []
            print("Please retake image")
        for j in range(0,len(imgPaths)) :                                                       ## Loop through each face and test if biggest face captured is recognized
            try :
                result = DeepFace.verify(img1_path = embeddings[j], 
                            img2_path = embedTest, 
                            distance_metric = metrics[1],
                            model_name = models[0],
                            silent = True
                )
                if result['verified'] == True :
                    adminTest = imgPaths[j]                                                     ## If any of the admin faces match, adminFace is updated
                    adminFace.append(adminTest[13:-4])
            except ValueError :                                                                 ## If no face is detected in frame return notice
                adminFace.append("No Face Detected")
        if len(adminFace)== 1 :
            adminFace.append("Stranger")                                                        ## if no admin faces match, adminFace is updated with Stranger
        faceRecognized = adminFace[1]
        adminFace = [""]                                                                        ## Reset adminFace for next capture
        print(faceRecognized)                                                                   ## Print name of face recognized
