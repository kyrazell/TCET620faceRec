## Import dependencies
from deepface import DeepFace
import numpy as np
import cv2
import fnmatch
import os

def faceName(
    imgLoc: str = "imgDataCount",
    adminNames: List[str],
    modelName: str = "VGG-Face",
    detectorBackend: str = "opencv",
    distanceMetric: str = "cosine",
) -> List[str]:
    """
    If a detected face matches admin face, return name of admin.

    The verification function converts facial images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        imgLoc (str or np.ndarray or  or List[float]): Path to the images.
            Accepts exact image path as a string.

        adminNames (List[str]): List of face names in same order as database.
        
        modelName (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detectorBackend (str): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv)

        distanceMetric (str): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

    Returns:
        result (List[str]): Returns names of faces recognized
    """


    ## Define image paths for all admin faces (Increasing pictures significantly affects computation time)
    imgLocI = imgLoc + "\I"

    imgPathsC = [imgLocI + '0.jpg']
    for i in range(1,len(fnmatch.filter(os.listdir(imgLoc), '*.jpg'))):
        imgPathsC.append(imgLocI + str(i) + ".jpg")

        imgPaths = imgPathsC[1:]
    
    ## Initialize list to identify if any of the faces captured in the frame match an admin face
    adminFace = [""]
    
    ## Initialize selected camera
    liveCam = cv2.VideoCapture(0)

    ## Run face verification against each face in admin database

    ret, frame = liveCam.read()                                 ## Test if cv2/camera supports multi-output (May interrupt live video output if camera 0 is seized
    testImg = "Test.jpg"                                        ## Define name for captured frame
    cv2.imwrite(testImg, frame)                                 ## Capture frame
    for j in range(0,len(imgPaths)):                            ## Loop through each admin face and test if any faces captured on frame match admin
        try:
            result = DeepFace.verify(img1_path = imgPaths[j], 
                      img2_path = "Test.jpg", 
                      distance_metric = distanceMetric,
                      model_name = modelName,
                      detector_backend = detectorBackend
            )
            if result['verified'] == True :
                adminFace.append(adminNames[j])                 ## If any of the admin faces match, adminFace is updated
        except ValueError:                                      ## If no face is detected in frame, do not crash
            adminFace.append("No Face Detected")
        
    if len(adminFace)== 1 :
        adminFace.append("Stranger")                            ## if no admin faces match, adminFace is updated with Stranger
    faceRecognized = adminFace[1:]
    return faceRecognized
