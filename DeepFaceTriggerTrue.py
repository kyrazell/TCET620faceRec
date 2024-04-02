## Import dependencies
from deepface import DeepFace
import cv2
import fnmatch
import os

def faceVerify(
    imgLoc: str = "imgDataCount",
    modelName: str = "VGG-Face",
    detectorBackend: str = "opencv",
    distanceMetric: str = "cosine",
) -> Dict[str, Any]:
    """
    Verify if a detected face matches admin faces.

    The verification function converts facial images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        imgLoc (str): Path to face images.
            Accepts exact image path as a string, all images in path must be .jpg files with filenames as ascending numbers from 0 i.e. "0.jpg" "1.jpg".

        modelName (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detectorBackend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv)

        distanceMetric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

    Returns:
        result (bool): Indicates whether any person in frame is an admin
    """


    ## Define image paths for all admin faces (Increasing pictures significantly affects computation time)
    imgPaths = [imgLoc + '\\0.jpg']
    for i in range(0,len(fnmatch.filter(os.listdir(imgLoc), '*.jpg'))):
        imgPaths.append(imgLoc + "\\" + str(i) + ".jpg")

    ## Initialize boolean matrix to identify if any of the faces captured in the frame match an admin face
    adminStatus = [False]

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
            adminStatus.append(result['verified'])              ## Print boolean verified result to adminStatus boolean array
        except ValueError:                                      ## If no face is detected in frame, return false
            adminStatus.append(False)
    adminVerify = max(adminStatus)                              ## if any of the admin faces match, adminVerify returns true (Check DeepFaceTriggerTrue.py)
            
    return adminVerify
